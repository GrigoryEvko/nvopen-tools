// Function: sub_254BB20
// Address: 0x254bb20
//
__int64 __fastcall sub_254BB20(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rsi
  __int64 result; // rax

  v2 = sub_250C3F0(a2, *a1);
  if ( !v2 )
    return 0;
  v3 = (_BYTE *)v2;
  result = 0;
  if ( *v3 == 17 )
  {
    sub_AE6EC0(a1[1], (__int64)v3);
    return 1;
  }
  return result;
}
