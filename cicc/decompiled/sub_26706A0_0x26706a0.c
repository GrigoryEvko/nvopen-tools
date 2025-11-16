// Function: sub_26706A0
// Address: 0x26706a0
//
void __fastcall sub_26706A0(_QWORD *a1)
{
  unsigned __int64 v1; // r14
  _QWORD *v2; // r13
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v1 = (unsigned __int64)(a1 - 11);
  v2 = a1 + 2;
  v4 = a1 + 22;
  *(a1 - 11) = off_4A1FD58;
  *a1 = &unk_4A1FDE8;
  do
  {
    v5 = *((unsigned int *)v4 - 2);
    v6 = *(v4 - 3);
    v4 -= 4;
    sub_C7D6A0(v6, 16 * v5, 8);
  }
  while ( v4 != v2 );
  v7 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v7 != a1 - 4 )
    _libc_free(v7);
  sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
  j_j___libc_free_0(v1);
}
