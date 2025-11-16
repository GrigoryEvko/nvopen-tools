// Function: sub_253A290
// Address: 0x253a290
//
__int64 __fastcall sub_253A290(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_4A171B8;
  v2 = *(_QWORD *)(a1 + 56);
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 32), 24LL * *(unsigned int *)(a1 + 48), 8);
}
