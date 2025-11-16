// Function: sub_2C88CF0
// Address: 0x2c88cf0
//
__int64 __fastcall sub_2C88CF0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rdi

  v2 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = off_49D3FA0;
  return sub_C7D6A0(v5, 8 * v4, 8);
}
