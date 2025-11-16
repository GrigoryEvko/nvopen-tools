// Function: sub_2670560
// Address: 0x2670560
//
__int64 __fastcall sub_2670560(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = a1 - 2;
  v3 = a1 + 18;
  *(a1 - 11) = off_4A1FD58;
  *a1 = &unk_4A1FDE8;
  do
  {
    v4 = *((unsigned int *)v3 + 6);
    v5 = v3[1];
    v3 -= 4;
    sub_C7D6A0(v5, 16 * v4, 8);
  }
  while ( v1 != v3 );
  v6 = *(a1 - 6);
  *(a1 - 11) = &unk_4A16C00;
  if ( (_QWORD *)v6 != a1 - 4 )
    _libc_free(v6);
  return sub_C7D6A0(*(a1 - 9), 8LL * *((unsigned int *)a1 - 14), 8);
}
