// Function: sub_E0C970
// Address: 0xe0c970
//
const char *__fastcall sub_E0C970(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "DW_LLE_end_of_list";
      break;
    case 1:
      result = "DW_LLE_base_addressx";
      break;
    case 2:
      result = "DW_LLE_startx_endx";
      break;
    case 3:
      result = "DW_LLE_startx_length";
      break;
    case 4:
      result = "DW_LLE_offset_pair";
      break;
    case 5:
      result = "DW_LLE_default_location";
      break;
    case 6:
      result = "DW_LLE_base_address";
      break;
    case 7:
      result = "DW_LLE_start_end";
      break;
    case 8:
      result = "DW_LLE_start_length";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
