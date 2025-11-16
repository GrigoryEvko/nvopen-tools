// Function: sub_1B8F6B0
// Address: 0x1b8f6b0
//
__int64 __fastcall sub_1B8F6B0(_QWORD *a1)
{
  unsigned __int64 *v2; // r12
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rdi

  v2 = a1 + 14;
  v3 = (unsigned __int64 *)a1[15];
  *a1 = &unk_49F7110;
  if ( a1 + 14 != v3 )
  {
    do
    {
      v4 = v3;
      v3 = (unsigned __int64 *)v3[1];
      v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
      *v3 = v5 | *v3 & 7;
      *(_QWORD *)(v5 + 8) = v3;
      v4[1] = 0;
      *v4 &= 7u;
      (*(void (__fastcall **)(unsigned __int64 *))(*(v4 - 1) + 8))(v4 - 1);
    }
    while ( v2 != v3 );
    v6 = (unsigned __int64 *)a1[15];
    while ( v2 != v6 )
    {
      v7 = v6;
      v6 = (unsigned __int64 *)v6[1];
      v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      *v6 = v8 | *v6 & 7;
      *(_QWORD *)(v8 + 8) = v6;
      v7[1] = 0;
      *v7 &= 7u;
      (*(void (__fastcall **)(unsigned __int64 *))(*(v7 - 1) + 8))(v7 - 1);
    }
  }
  v9 = a1[10];
  *a1 = &unk_49F6D50;
  if ( (_QWORD *)v9 != a1 + 12 )
    _libc_free(v9);
  v10 = a1[7];
  if ( (_QWORD *)v10 != a1 + 9 )
    _libc_free(v10);
  v11 = (_QWORD *)a1[2];
  if ( v11 != a1 + 4 )
    j_j___libc_free_0(v11, a1[4] + 1LL);
  return j_j___libc_free_0(a1, 128);
}
