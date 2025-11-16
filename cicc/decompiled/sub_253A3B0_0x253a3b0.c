// Function: sub_253A3B0
// Address: 0x253a3b0
//
void __fastcall sub_253A3B0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_4A171B8;
  v2 = *(_QWORD *)(a1 + 56);
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 24LL * *(unsigned int *)(a1 + 48), 8);
  j_j___libc_free_0(a1);
}
