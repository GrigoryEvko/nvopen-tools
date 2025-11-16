// Function: sub_2C76340
// Address: 0x2c76340
//
void __fastcall sub_2C76340(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A25050;
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 8LL * *(unsigned int *)(a1 + 352), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), 8LL * *(unsigned int *)(a1 + 320), 8);
  *(_QWORD *)(a1 + 240) = &unk_49DD210;
  sub_CB5840(a1 + 240);
  v2 = *(_QWORD *)(a1 + 208);
  if ( v2 != a1 + 224 )
    j_j___libc_free_0(v2);
  sub_BB9260(a1);
  j_j___libc_free_0(a1);
}
