// Function: sub_BCAC60
// Address: 0xbcac60
//
__int64 __fastcall sub_BCAC60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
      result = sub_C332F0(a1, a2, a3, a4);
      break;
    case 1:
      result = sub_C33300();
      break;
    case 2:
      result = sub_C33310(a1, a2);
      break;
    case 3:
      result = sub_C33320(a1);
      break;
    case 4:
      result = sub_C33420();
      break;
    case 5:
      result = sub_C33330();
      break;
    case 6:
      result = sub_C33340(a1, a2, a3, a4, a5);
      break;
    default:
      BUG();
  }
  return result;
}
