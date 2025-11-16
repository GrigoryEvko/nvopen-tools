// Function: sub_160DAC0
// Address: 0x160dac0
//
__int64 __fastcall sub_160DAC0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *a1 = &unk_49ED650;
  v2 = a1[23];
  if ( v2 )
    j_j___libc_free_0(v2, a1[25] - v2);
  v3 = a1[20];
  if ( v3 )
    j_j___libc_free_0(v3, a1[22] - v3);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
  return j_j___libc_free_0(a1, 216);
}
