// Function: sub_D00830
// Address: 0xd00830
//
__int64 __fastcall sub_D00830(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13

  v2 = a1[22];
  *a1 = &unk_49DDC38;
  if ( v2 )
  {
    if ( !*(_BYTE *)(v2 + 68) )
      _libc_free(*(_QWORD *)(v2 + 48), a2);
    j_j___libc_free_0(v2, 200);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
