// Function: sub_302FC90
// Address: 0x302fc90
//
void __fastcall sub_302FC90(unsigned __int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi

  v2 = *(_QWORD *)(a1 + 525256);
  *(_QWORD *)a1 = &unk_4A2CC60;
  while ( v2 )
  {
    sub_302FA60(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 8LL * *(unsigned int *)(a1 + 48), 4);
  j_j___libc_free_0(a1);
}
