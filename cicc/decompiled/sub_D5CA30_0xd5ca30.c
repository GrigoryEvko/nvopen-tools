// Function: sub_D5CA30
// Address: 0xd5ca30
//
char *__fastcall sub_D5CA30(_DWORD *a1)
{
  char *result; // rax

  switch ( *a1 )
  {
    case 0:
      result = "malloc";
      break;
    case 1:
      result = "_Znwm";
      break;
    case 2:
      result = "_ZnwmSt11align_val_t";
      break;
    case 3:
      result = "_Znam";
      break;
    case 4:
      result = "_ZnamSt11align_val_t";
      break;
    case 5:
      result = "??2@YAPAXI@Z";
      break;
    case 6:
      result = "??_U@YAPAXI@Z";
      break;
    case 7:
      result = "vec_malloc";
      break;
    case 8:
      result = "__kmpc_alloc_shared";
      break;
    default:
      BUG();
  }
  return result;
}
