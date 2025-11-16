// Function: sub_2C88E30
// Address: 0x2c88e30
//
void __fastcall sub_2C88E30(unsigned __int64 a1)
{
  unsigned __int64 *v1; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rdi

  v1 = *(unsigned __int64 **)(a1 + 48);
  *(_QWORD *)a1 = off_49D4000;
  if ( v1 )
  {
    if ( (unsigned __int64 *)*v1 != v1 + 2 )
      _libc_free(*v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 88), 8LL * *(unsigned int *)(a1 + 104), 8);
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != a1 + 72 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = off_49D3FA0;
  sub_C7D6A0(v5, 8 * v4, 8);
  j_j___libc_free_0(a1);
}
