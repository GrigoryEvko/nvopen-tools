// Function: sub_28E5B20
// Address: 0x28e5b20
//
void __fastcall sub_28E5B20(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A21C58;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
