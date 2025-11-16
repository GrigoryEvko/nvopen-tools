// Function: sub_2305EE0
// Address: 0x2305ee0
//
void __fastcall sub_2305EE0(unsigned __int64 a1)
{
  __int64 v1; // rsi
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi

  v1 = *(unsigned int *)(a1 + 88);
  *(_QWORD *)a1 = &unk_4A0B2B8;
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16 * v1, 8);
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    if ( v4 )
      j_j___libc_free_0(v4);
    j_j___libc_free_0(v3);
  }
  j_j___libc_free_0(a1);
}
