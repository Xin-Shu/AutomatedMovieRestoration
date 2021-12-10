close all
clear

org_folder = "M:\MAI_dataset\Origin_set\BBB-360-png";
degrade_folder = "M:\MAI_dataset\Degraded_set\BBB\frame";
mask_folder = "M:\MAI_dataset\Degraded_set\BBB\mask";

if ~isfolder(org_folder)
  errorMessage = sprintf( ...
      'Error: The following folder does not exist:\n%s', org_folder);
  uiwait(warndlg(errorMessage));
  return;
end

imgPattern = fullfile(org_folder, '*.png');
pngFiles = dir(imgPattern);

% Processing images
max_width = 3;
colormap(gray(256));

for i = 1 : length(pngFiles)
    frameName = pngFiles(i).name;
    if mod(i, 10) == 0
        fprintf("Processing: %d of %d -- '%s'.\n", ...
            i, length(pngFiles), frameName)
    end
    fullName = fullfile(org_folder, frameName);
    frame_org = imread(fullName);
    gray_frame = im2gray(imresize(frame_org, [180, 320])); % 1/5 of 1080p
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
            right_boundary = line_pos - w;
        end
        scratch_width = right_boundary- left_boundary + 1;
        
        binary_mask(:, left_boundary : right_boundary)= ...
            ones(180, scratch_width);

        slope = randi([-10, 10]) * 0.0005;
        for j = 1 : rows
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
    imwrite(degrade2(1:180, 1:320),...
        colormap(gray(256)), degradedFullName); 
    imwrite(binary_mask(1:180, 1:320), maskFullName); 
    % imshow(imresize(degrade2, [800, 1920]), colormap(gray(256)));
    
end

fprintf('%s\n', "INFO: Finished Images Processing!");

function profile = makeLineProfile(cols, pos, amplitude, damping, m, row, w)
    x = 1 : cols;
    dx = abs(x - (m * row + pos));
    profile = amplitude * (damping .^ (dx)) .* cos(3 * pi * dx / (2 * w));
end