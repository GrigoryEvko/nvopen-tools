// Function: sub_254FD20
// Address: 0x254fd20
//
__int64 __fastcall sub_254FD20(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != v2 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 8LL * *(unsigned int *)(a1 + 24), 8);
}
