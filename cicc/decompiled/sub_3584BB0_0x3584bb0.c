// Function: sub_3584BB0
// Address: 0x3584bb0
//
__int64 __fastcall sub_3584BB0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v3; // rdi

  v1 = (_QWORD *)a1[32];
  *a1 = &unk_4A39928;
  if ( v1 )
  {
    *v1 = &unk_4A39900;
    sub_3584890((__int64)v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  v3 = a1[26];
  if ( (_QWORD *)v3 != a1 + 28 )
    j_j___libc_free_0(v3);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
