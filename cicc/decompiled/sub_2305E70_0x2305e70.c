// Function: sub_2305E70
// Address: 0x2305e70
//
void __fastcall sub_2305E70(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi

  v1 = *(unsigned int *)(a1 + 88);
  *(_QWORD *)a1 = &unk_4A0B2B8;
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16 * v1, 8);
  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    if ( v3 )
      j_j___libc_free_0(v3);
    j_j___libc_free_0(v2);
  }
}
