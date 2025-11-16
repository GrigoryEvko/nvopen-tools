// Function: sub_13FB670
// Address: 0x13fb670
//
__int64 __fastcall sub_13FB670(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdi

  v2 = a1 + 160;
  *(_QWORD *)(v2 - 160) = &unk_49EABD8;
  sub_13FB2B0(v2);
  v3 = *(unsigned __int64 **)(a1 + 232);
  v4 = &v3[*(unsigned int *)(a1 + 240)];
  while ( v4 != v3 )
  {
    v5 = *v3++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 280);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 288)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 280);
  }
  if ( v7 != a1 + 296 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 232);
  if ( v9 != a1 + 248 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 192);
  if ( v10 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 208) - v10);
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
