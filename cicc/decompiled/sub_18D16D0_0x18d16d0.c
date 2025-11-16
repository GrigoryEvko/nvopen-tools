// Function: sub_18D16D0
// Address: 0x18d16d0
//
__int64 *__fastcall sub_18D16D0(__int64 *a1, char *a2, __int64 *a3)
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
  __int64 v15; // rcx
  __int16 v16; // dx
  __int64 v17; // rdx
  char *v18; // r12
  __int64 i; // r15
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rdi
  char *j; // r13
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 *v29; // [rsp+0h] [rbp-60h]
  __int64 *v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  char *v34; // [rsp+28h] [rbp-38h]

  v4 = (char *)a1[1];
  v34 = (char *)*a1;
  v5 = 0x86BCA1AF286BCA1BLL * ((__int64)&v4[-*a1] >> 3);
  if ( v5 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0x86BCA1AF286BCA1BLL * ((__int64)&v4[-*a1] >> 3);
  v7 = a2;
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x79435E50D79435E5LL * ((__int64)&v4[-*a1] >> 3);
  v10 = a2 - v34;
  if ( v8 )
  {
    v27 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v9 )
    {
      v31 = 0;
      v11 = 152;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0xD79435E50D7943LL )
      v9 = 0xD79435E50D7943LL;
    v27 = 152 * v9;
  }
  v29 = a3;
  v28 = sub_22077B0(v27);
  v10 = a2 - v34;
  a3 = v29;
  v33 = v28;
  v31 = v28 + v27;
  v11 = v28 + 152;
LABEL_7:
  v12 = v33 + v10 == 0;
  v13 = v33 + v10;
  v14 = v13;
  if ( !v12 )
  {
    v15 = *a3;
    v30 = a3;
    *(_BYTE *)(v13 + 10) = *((_BYTE *)a3 + 10);
    v16 = *((_WORD *)a3 + 8);
    *(_QWORD *)v13 = v15;
    LOWORD(v15) = *((_WORD *)a3 + 4);
    *(_WORD *)(v13 + 16) = v16;
    v17 = a3[3];
    *(_WORD *)(v13 + 8) = v15;
    *(_QWORD *)(v13 + 24) = v17;
    sub_16CCEE0((_QWORD *)(v13 + 32), v13 + 72, 2, (__int64)(a3 + 4));
    sub_16CCEE0((_QWORD *)(v14 + 88), v14 + 128, 2, (__int64)(v30 + 11));
    *(_BYTE *)(v14 + 144) = *((_BYTE *)v30 + 144);
  }
  v18 = v34;
  if ( a2 != v34 )
  {
    for ( i = v33; ; i += 152 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v18;
        *(_BYTE *)(i + 8) = v18[8];
        *(_BYTE *)(i + 9) = v18[9];
        *(_BYTE *)(i + 10) = v18[10];
        *(_BYTE *)(i + 16) = v18[16];
        *(_BYTE *)(i + 17) = v18[17];
        *(_QWORD *)(i + 24) = *((_QWORD *)v18 + 3);
        sub_16CCCB0((_QWORD *)(i + 32), i + 72, (__int64)(v18 + 32));
        sub_16CCCB0((_QWORD *)(i + 88), i + 128, (__int64)(v18 + 88));
        *(_BYTE *)(i + 144) = v18[144];
      }
      v18 += 152;
      if ( a2 == v18 )
        break;
    }
    v11 = i + 304;
  }
  if ( a2 != v4 )
  {
    do
    {
      *(_QWORD *)v11 = *(_QWORD *)v7;
      *(_BYTE *)(v11 + 8) = v7[8];
      *(_BYTE *)(v11 + 9) = v7[9];
      *(_BYTE *)(v11 + 10) = v7[10];
      *(_BYTE *)(v11 + 16) = v7[16];
      *(_BYTE *)(v11 + 17) = v7[17];
      *(_QWORD *)(v11 + 24) = *((_QWORD *)v7 + 3);
      sub_16CCCB0((_QWORD *)(v11 + 32), v11 + 72, (__int64)(v7 + 32));
      v20 = (__int64)(v7 + 88);
      v21 = v11 + 128;
      v7 += 152;
      v22 = (_QWORD *)(v11 + 88);
      v11 += 152;
      sub_16CCCB0(v22, v21, v20);
      *(_BYTE *)(v11 - 8) = *(v7 - 8);
    }
    while ( v4 != v7 );
  }
  for ( j = v34; j != v4; j += 152 )
  {
    v24 = *((_QWORD *)j + 13);
    if ( v24 != *((_QWORD *)j + 12) )
      _libc_free(v24);
    v25 = *((_QWORD *)j + 6);
    if ( v25 != *((_QWORD *)j + 5) )
      _libc_free(v25);
  }
  if ( v34 )
    j_j___libc_free_0(v34, a1[2] - (_QWORD)v34);
  *a1 = v33;
  a1[1] = v11;
  a1[2] = v31;
  return a1;
}
