// Function: sub_18D1050
// Address: 0x18d1050
//
__int64 *__fastcall sub_18D1050(__int64 *a1, char *a2, __int64 *a3)
{
  char *v4; // r14
  char *v6; // rbx
  char *v7; // rcx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // r10
  __int64 v13; // r15
  bool v14; // zf
  __int64 v15; // r10
  __int64 v16; // r12
  __int64 v17; // rsi
  __int16 v18; // dx
  __int64 v19; // rdx
  char *v20; // r12
  __int64 i; // r15
  __int64 v22; // rdx
  __int64 v23; // rsi
  _QWORD *v24; // rdi
  char *j; // r13
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 *v31; // [rsp+0h] [rbp-60h]
  __int64 *v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  char *v36; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v6 = (char *)a1[1];
  v7 = (char *)*a1;
  v36 = (char *)*a1;
  v8 = 0x86BCA1AF286BCA1BLL * ((__int64)&v6[-*a1] >> 3);
  if ( v8 == 0xD79435E50D7943LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0x86BCA1AF286BCA1BLL * ((v6 - v7) >> 3);
  v10 = __CFADD__(v9, v8);
  v11 = v9 - 0x79435E50D79435E5LL * ((v6 - v7) >> 3);
  v12 = a2 - v36;
  if ( v10 )
  {
    v29 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v11 )
    {
      v33 = 0;
      v13 = 152;
      v35 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0xD79435E50D7943LL )
      v11 = 0xD79435E50D7943LL;
    v29 = 152 * v11;
  }
  v31 = a3;
  v30 = sub_22077B0(v29);
  v12 = a2 - v36;
  a3 = v31;
  v35 = v30;
  v33 = v30 + v29;
  v13 = v30 + 152;
LABEL_7:
  v14 = v35 + v12 == 0;
  v15 = v35 + v12;
  v16 = v15;
  if ( !v14 )
  {
    v17 = *a3;
    v32 = a3;
    *(_BYTE *)(v15 + 10) = *((_BYTE *)a3 + 10);
    v18 = *((_WORD *)a3 + 8);
    *(_QWORD *)v15 = v17;
    LOWORD(v17) = *((_WORD *)a3 + 4);
    *(_WORD *)(v15 + 16) = v18;
    v19 = a3[3];
    *(_WORD *)(v15 + 8) = v17;
    *(_QWORD *)(v15 + 24) = v19;
    sub_16CCCB0((_QWORD *)(v15 + 32), v15 + 72, (__int64)(a3 + 4));
    sub_16CCCB0((_QWORD *)(v16 + 88), v16 + 128, (__int64)(v32 + 11));
    *(_BYTE *)(v16 + 144) = *((_BYTE *)v32 + 144);
  }
  v20 = v36;
  if ( a2 != v36 )
  {
    for ( i = v35; ; i += 152 )
    {
      if ( i )
      {
        *(_QWORD *)i = *(_QWORD *)v20;
        *(_BYTE *)(i + 8) = v20[8];
        *(_BYTE *)(i + 9) = v20[9];
        *(_BYTE *)(i + 10) = v20[10];
        *(_BYTE *)(i + 16) = v20[16];
        *(_BYTE *)(i + 17) = v20[17];
        *(_QWORD *)(i + 24) = *((_QWORD *)v20 + 3);
        sub_16CCCB0((_QWORD *)(i + 32), i + 72, (__int64)(v20 + 32));
        sub_16CCCB0((_QWORD *)(i + 88), i + 128, (__int64)(v20 + 88));
        *(_BYTE *)(i + 144) = v20[144];
      }
      v20 += 152;
      if ( a2 == v20 )
        break;
    }
    v13 = i + 304;
  }
  if ( a2 != v6 )
  {
    do
    {
      *(_QWORD *)v13 = *(_QWORD *)v4;
      *(_BYTE *)(v13 + 8) = v4[8];
      *(_BYTE *)(v13 + 9) = v4[9];
      *(_BYTE *)(v13 + 10) = v4[10];
      *(_BYTE *)(v13 + 16) = v4[16];
      *(_BYTE *)(v13 + 17) = v4[17];
      *(_QWORD *)(v13 + 24) = *((_QWORD *)v4 + 3);
      sub_16CCCB0((_QWORD *)(v13 + 32), v13 + 72, (__int64)(v4 + 32));
      v22 = (__int64)(v4 + 88);
      v23 = v13 + 128;
      v4 += 152;
      v24 = (_QWORD *)(v13 + 88);
      v13 += 152;
      sub_16CCCB0(v24, v23, v22);
      *(_BYTE *)(v13 - 8) = *(v4 - 8);
    }
    while ( v6 != v4 );
  }
  for ( j = v36; j != v6; j += 152 )
  {
    v26 = *((_QWORD *)j + 13);
    if ( v26 != *((_QWORD *)j + 12) )
      _libc_free(v26);
    v27 = *((_QWORD *)j + 6);
    if ( v27 != *((_QWORD *)j + 5) )
      _libc_free(v27);
  }
  if ( v36 )
    j_j___libc_free_0(v36, a1[2] - (_QWORD)v36);
  *a1 = v35;
  a1[1] = v13;
  a1[2] = v33;
  return a1;
}
