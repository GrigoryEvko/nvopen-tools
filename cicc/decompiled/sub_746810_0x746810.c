// Function: sub_746810
// Address: 0x746810
//
const char *__fastcall sub_746810(char a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 5:
      result = "__underlying_type";
      break;
    case 13:
      result = "__add_lvalue_reference";
      break;
    case 14:
      result = "__add_pointer";
      break;
    case 15:
      result = "__add_rvalue_reference";
      break;
    case 16:
      result = "__decay";
      break;
    case 17:
      result = "__make_signed";
      break;
    case 18:
      result = "__make_unsigned";
      break;
    case 19:
      result = "__remove_all_extents";
      break;
    case 20:
      result = "__remove_const";
      break;
    case 21:
      result = "__remove_cv";
      break;
    case 22:
      result = "__remove_cvref";
      break;
    case 23:
      result = "__remove_extent";
      break;
    case 24:
      result = "__remove_pointer";
      break;
    case 25:
      result = "__remove_reference_t";
      break;
    case 26:
      result = "__remove_restrict";
      break;
    case 27:
      result = "__remove_volatile";
      break;
    case 28:
      result = "__remove_reference";
      break;
    default:
      sub_721090();
  }
  return result;
}
