// Function: sub_195E400
// Address: 0x195e400
//
__int64 __fastcall sub_195E400(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = a1[9];
  *a1 = off_49F3B68;
  if ( v1 )
    sub_161E7C0((__int64)(a1 + 9), v1);
  return j_j___libc_free_0(a1, 112);
}
