close all
clear

org_folder = "DetectorTrainingModel\dataset\SintelTrailer_original";
degrade_folder = "DetectorTrainingModel\dataset\320_180\SintelTrailer_degraded";
mask_folder = "DetectorTrainingModel\dataset\320_180\SintelTrailer_BinaryMask";

if ~isfolder(org_folder)
  errorMessage = sprintf( ...
      'Error: The following folder does not exist:\n%s', org_folder);
  uiwait(warndlg(errorMessage));
  return;
end

imgPattern = fullfile(org_folder, '*.png');
pngFiles = dir(imgPattern);
topPixel = 22;
botPixel = 158;

% Processing images
max_width = 3;
colormap(gray(256));

for i = 1 : length(pngFiles)
    frameName = pngFiles(i).name;
    fullName = fullfile(org_folder, frameName);
    frame_org = imread(fullName);
    gray_frame = im2gray(imresize(frame_org, 1/4));
    [rows, cols, chan] = size(gray_frame);
    binary_mask = zeros(180, 320, 1, 'double');
    degrade = gray_frame;
    degrade2 = double(degrade);
    
    for line_num = 1 : (1 + floor(rand * 5)) % Add randomly up to 5 lines
        line_pos = floor(rand * (cols - 2 * max_width)) + max_width + 1;
        w = round(1 + rand * max_width);
        a = rand * 100;

        if (line_pos - w > 1)
            left_boundary = line_pos - w;
        else
            left_boundary = 1;
        end
        if (line_pos + w <= 320)
            right_boundary = line_pos + w;
        else
            right_boundary = 210;
        end
        scratch_width = right_boundary- left_boundary + 1;
        
        binary_mask(topPixel : botPixel, left_boundary : right_boundary)= ...
            ones((botPixel - topPixel + 1), scratch_width);

        slope = randi([-10, 10]) * 0.0005;
        for j = topPixel : botPixel
            profile = makeLineProfile(...
                cols, line_pos, (a - 50), 0.25, slope, j, w);
            degrade2(j, left_boundary : right_boundary) = ...
                double(degrade2(j, left_boundary : right_boundary)) ...
                + profile(:, left_boundary : right_boundary);
        end
    end
    
    % binary_mask = binary_mask > 0;
    degradedFullName = fullfile(degrade_folder, frameName);
    maskFullName = fullfile(mask_folder, frameName);
    imwrite(degrade2(topPixel+1:botPixel, 1:320),...
        colormap(gray(256)), degradedFullName); 
    imwrite(binary_mask(topPixel+1:botPixel, 1:320), maskFullName); 
    % imshow(imresize(degrade2, [1080, 1920]), colormap(gray(256)));
    
end

fprintf('%s\n', "INFO: Finished Images Processing!");

function profile = makeLineProfile(cols, pos, amplitude, damping, m, row, w)
    x = 1 : cols;
    dx = abs(x - (m * row + pos));
    profile = amplitude * (damping .^ (dx)) .* cos(3 * pi * dx / (2 * w));
end