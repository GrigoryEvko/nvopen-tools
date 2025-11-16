// Function: sub_2FE45C0
// Address: 0x2fe45c0
//
__int64 __fastcall sub_2FE45C0(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi

  v2 = *(_QWORD *)(a1 + 525256);
  *(_QWORD *)a1 = &unk_4A2CC60;
  while ( v2 )
  {
    sub_2FE43F0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 32), 8LL * *(unsigned int *)(a1 + 48), 4);
}
