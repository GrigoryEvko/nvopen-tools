// Function: sub_26704C0
// Address: 0x26704c0
//
__int64 __fastcall sub_26704C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = a1 + 72;
  v3 = a1 + 232;
  *(_QWORD *)a1 = off_4A1FD58;
  *(_QWORD *)(a1 + 88) = &unk_4A1FDE8;
  do
  {
    v4 = *(unsigned int *)(v3 + 24);
    v5 = *(_QWORD *)(v3 + 8);
    v3 -= 32;
    sub_C7D6A0(v5, 16 * v4, 8);
  }
  while ( v1 != v3 );
  v6 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v6 != a1 + 56 )
    _libc_free(v6);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
