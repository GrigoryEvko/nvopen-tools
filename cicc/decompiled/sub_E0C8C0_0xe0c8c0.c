// Function: sub_E0C8C0
// Address: 0xe0c8c0
//
const char *__fastcall sub_E0C8C0(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "DW_RLE_end_of_list";
      break;
    case 1:
      result = "DW_RLE_base_addressx";
      break;
    case 2:
      result = "DW_RLE_startx_endx";
      break;
    case 3:
      result = "DW_RLE_startx_length";
      break;
    case 4:
      result = "DW_RLE_offset_pair";
      break;
    case 5:
      result = "DW_RLE_base_address";
      break;
    case 6:
      result = "DW_RLE_start_end";
      break;
    case 7:
      result = "DW_RLE_start_length";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
