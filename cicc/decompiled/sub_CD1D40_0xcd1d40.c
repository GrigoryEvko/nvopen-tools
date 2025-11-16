// Function: sub_CD1D40
// Address: 0xcd1d40
//
__int64 __fastcall sub_CD1D40(__int64 *a1, __int64 a2, char a3)
{
  __int64 v3; // r8
  __int64 result; // rax

  v3 = *a1;
  a1[10] += a2;
  result = (v3 + (1LL << a3) - 1) & -(1LL << a3);
  if ( a1[1] < (unsigned __int64)(a2 + result) || !v3 )
    return sub_9D1E70((__int64)a1, a2, a2, a3);
  *a1 = a2 + result;
  return result;
}
