// Function: sub_B12320
// Address: 0xb12320
//
__int64 __fastcall sub_B12320(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 32);
  if ( v1 )
  {
    if ( v1 != 1 )
      BUG();
    if ( *(_QWORD *)(a1 + 40) )
      sub_B91220(a1 + 40);
    if ( *(_QWORD *)(a1 + 24) )
      sub_B91220(a1 + 24);
    return j_j___libc_free_0(a1, 48);
  }
  else
  {
    if ( *(_QWORD *)(a1 + 88) )
      sub_B91220(a1 + 88);
    if ( *(_QWORD *)(a1 + 80) )
      sub_B91220(a1 + 80);
    if ( *(_QWORD *)(a1 + 72) )
      sub_B91220(a1 + 72);
    sub_B91360(a1 + 40);
    if ( *(_QWORD *)(a1 + 24) )
      sub_B91220(a1 + 24);
    return j_j___libc_free_0(a1, 96);
  }
}
