// Function: sub_ADBE50
// Address: 0xadbe50
//
__int64 __fastcall sub_ADBE50(__int64 *a1, __int64 *a2, _BYTE *a3)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rsi

  switch ( *(_BYTE *)a1 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      sub_4089E2();
    case 4:
      result = sub_ACBDD0((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    case 5:
      result = sub_ADBCA0((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    case 6:
      result = sub_AD4CC0((unsigned __int64)a1, (__int64)a2, a3);
      v5 = result;
      break;
    case 7:
      result = sub_AD51C0((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    case 8:
      result = sub_AD0A60((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    case 9:
      result = sub_AD1A10((__int64)a1, a2, (__int64)a3);
      v5 = result;
      break;
    case 0xA:
      result = sub_AD2CE0((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    case 0xB:
      result = sub_AD3E30((__int64)a1, (__int64)a2, (__int64)a3);
      v5 = result;
      break;
    default:
      BUG();
  }
  if ( v5 )
  {
    sub_BD84D0(a1, v5);
    return sub_ACFDF0(a1, v5, v3);
  }
  return result;
}
