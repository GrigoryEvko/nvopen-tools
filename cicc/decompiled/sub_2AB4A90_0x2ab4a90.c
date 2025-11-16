// Function: sub_2AB4A90
// Address: 0x2ab4a90
//
void __fastcall sub_2AB4A90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != v2 )
    _libc_free(v3);
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 24), 4);
}
