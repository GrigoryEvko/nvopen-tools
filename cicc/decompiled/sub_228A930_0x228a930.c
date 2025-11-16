// Function: sub_228A930
// Address: 0x228a930
//
__int64 __fastcall sub_228A930(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A08D28;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
