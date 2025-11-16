// Function: sub_18D1E40
// Address: 0x18d1e40
//
__int64 *__fastcall sub_18D1E40(__int64 *a1, char *a2, __int64 a3)
{
  char *v4; // rbx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  char *v7; // r14
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // r11
  __int64 v11; // r15
  bool v12; // zf
  __int64 v13; // r11
  __int64 v14; // r12
  char *v15; // r12
  __int64 i; // r15
  __int64 v17; // rdx
  __int64 v18; // rsi
  _QWORD *v19; // rdi
  char *j; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  char *v31; // [rsp+28h] [rbp-38h]

  v4 = (char *)a1[1];
  v31 = (char *)*a1;
  v5 = 0x8E38E38E38E38E39LL * ((__int64)&v4[-*a1] >> 4);
  if ( v5 == 0xE38E38E38E38E3LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0x8E38E38E38E38E39LL * ((__int64)&v4[-*a1] >> 4);
  v7 = a2;
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x71C71C71C71C71C7LL * ((__int64)&v4[-*a1] >> 4);
  v10 = a2 - v31;
  if ( v8 )
  {
    v24 = 0x7FFFFFFFFFFFFFB0LL;
  }
  else
  {
    if ( !v9 )
    {
      v28 = 0;
      v11 = 144;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xE38E38E38E38E3LL )
      v9 = 0xE38E38E38E38E3LL;
    v24 = 144 * v9;
  }
  v26 = a3;
  v25 = sub_22077B0(v24);
  v10 = a2 - v31;
  a3 = v26;
  v30 = v25;
  v28 = v25 + v24;
  v11 = v25 + 144;
LABEL_7:
  v12 = v30 + v10 == 0;
  v13 = v30 + v10;
  v14 = v13;
  if ( !v12 )
  {
    v27 = a3;
    *(_QWORD *)v13 = *(_QWORD *)a3;
    *(_WORD *)(v13 + 8) = *(_WORD *)(a3 + 8);
    *(_QWORD *)(v13 + 16) = *(_QWORD *)(a3 + 16);
    sub_16CCEE0((_QWORD *)(v13 + 24), v13 + 64, 2, a3 + 24);
    sub_16CCEE0((_QWORD *)(v14 + 80), v14 + 120, 2, v27 + 80);
    *(_BYTE *)(v14 + 136) = *(_BYTE *)(v27 + 136);
  }
  v15 = v31;
  if ( a2 != v31 )
  {
    for ( i = v30; ; i += 144 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v15;
        *(_BYTE *)(i + 8) = v15[8];
        *(_BYTE *)(i + 9) = v15[9];
        *(_QWORD *)(i + 16) = *((_QWORD *)v15 + 2);
        sub_16CCCB0((_QWORD *)(i + 24), i + 64, (__int64)(v15 + 24));
        sub_16CCCB0((_QWORD *)(i + 80), i + 120, (__int64)(v15 + 80));
        *(_BYTE *)(i + 136) = v15[136];
      }
      v15 += 144;
      if ( a2 == v15 )
        break;
    }
    v11 = i + 288;
  }
  if ( a2 != v4 )
  {
    do
    {
      *(_QWORD *)v11 = *(_QWORD *)v7;
      *(_BYTE *)(v11 + 8) = v7[8];
      *(_BYTE *)(v11 + 9) = v7[9];
      *(_QWORD *)(v11 + 16) = *((_QWORD *)v7 + 2);
      sub_16CCCB0((_QWORD *)(v11 + 24), v11 + 64, (__int64)(v7 + 24));
      v17 = (__int64)(v7 + 80);
      v18 = v11 + 120;
      v7 += 144;
      v19 = (_QWORD *)(v11 + 80);
      v11 += 144;
      sub_16CCCB0(v19, v18, v17);
      *(_BYTE *)(v11 - 8) = *(v7 - 8);
    }
    while ( v4 != v7 );
  }
  for ( j = v31; j != v4; j += 144 )
  {
    v21 = *((_QWORD *)j + 12);
    if ( v21 != *((_QWORD *)j + 11) )
      _libc_free(v21);
    v22 = *((_QWORD *)j + 5);
    if ( v22 != *((_QWORD *)j + 4) )
      _libc_free(v22);
  }
  if ( v31 )
    j_j___libc_free_0(v31, a1[2] - (_QWORD)v31);
  *a1 = v30;
  a1[1] = v11;
  a1[2] = v28;
  return a1;
}
