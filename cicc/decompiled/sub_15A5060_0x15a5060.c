// Function: sub_15A5060
// Address: 0x15a5060
//
__int64 __fastcall sub_15A5060(__int64 a1, _BYTE *a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  __int64 result; // rax
  __int64 v8; // rsi

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      sub_41A076();
    case 4:
      result = sub_159B790(a1, (__int64)a2, a3);
      v8 = result;
      break;
    case 5:
      result = sub_15A4ED0(a1, (__int64)a2, a3, a5, a6, a7);
      v8 = result;
      break;
    case 6:
      result = sub_159E800(a1, (__int64)a2, a3, a4);
      v8 = result;
      break;
    case 7:
      result = sub_159F970(a1, a2, a3, (__int64)a4);
      v8 = result;
      break;
    case 8:
      result = sub_15A23E0(a1, (__int64)a2, a3);
      v8 = result;
      break;
  }
  if ( v8 )
  {
    sub_164D160(a1, v8);
    return sub_159D850(a1);
  }
  return result;
}
