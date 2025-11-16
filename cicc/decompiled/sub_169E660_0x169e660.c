// Function: sub_169E660
// Address: 0x169e660
//
__int64 __fastcall sub_169E660(__int64 a1, void *a2, char *a3, __int64 a4)
{
  _QWORD *v7; // rdi

  v7 = (_QWORD *)(a1 + 8);
  if ( a2 == sub_16982C0() )
    sub_169C4E0(v7, (__int64)a2);
  else
    sub_1698360((__int64)v7, (__int64)a2);
  return sub_169E610(a1, a3, a4, 0);
}
