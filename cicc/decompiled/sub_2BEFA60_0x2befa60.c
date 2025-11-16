// Function: sub_2BEFA60
// Address: 0x2befa60
//
void __fastcall sub_2BEFA60(_QWORD *a1)
{
  unsigned __int64 *v1; // r13
  unsigned __int64 *v3; // rax
  unsigned __int64 *v4; // rsi
  unsigned __int64 v5; // rcx
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v1 = a1 + 14;
  *a1 = &unk_4A23A00;
  v3 = (unsigned __int64 *)(a1[14] & 0xFFFFFFFFFFFFFFF8LL);
  if ( a1 + 14 != v3 )
  {
    do
    {
      v4 = (unsigned __int64 *)v3[1];
      v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      *v4 = v5 | *v4 & 7;
      *(_QWORD *)(v5 + 8) = v4;
      v3[1] = 0;
      *v3 &= 7u;
      (*(void (__fastcall **)(unsigned __int64 *))(*(v3 - 3) + 8))(v3 - 3);
      v3 = (unsigned __int64 *)(a1[14] & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( v3 != v1 );
  }
  v6 = (unsigned __int64 *)a1[15];
  while ( v6 != v1 )
  {
    v7 = v6;
    v6 = (unsigned __int64 *)v6[1];
    v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
    *v6 = v8 | *v6 & 7;
    *(_QWORD *)(v8 + 8) = v6;
    v7[1] = 0;
    *v7 &= 7u;
    (*(void (__fastcall **)(unsigned __int64 *))(*(v7 - 3) + 8))(v7 - 3);
  }
  v9 = a1[10];
  *a1 = &unk_4A23970;
  if ( (_QWORD *)v9 != a1 + 12 )
    _libc_free(v9);
  v10 = a1[7];
  if ( (_QWORD *)v10 != a1 + 9 )
    _libc_free(v10);
  v11 = a1[2];
  if ( (_QWORD *)v11 != a1 + 4 )
    j_j___libc_free_0(v11);
  j_j___libc_free_0((unsigned __int64)a1);
}
