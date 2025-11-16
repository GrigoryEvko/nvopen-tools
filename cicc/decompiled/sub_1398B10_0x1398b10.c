// Function: sub_1398B10
// Address: 0x1398b10
//
__int64 __fastcall sub_1398B10(char **a1, char *a2, _QWORD *a3, _QWORD *a4)
{
  char *v4; // r15
  char *v5; // r13
  __int64 v6; // rax
  char *v7; // r8
  char *v8; // r12
  bool v9; // zf
  __int64 v11; // rdi
  __int64 v12; // rax
  bool v13; // cf
  unsigned __int64 v14; // rax
  signed __int64 v15; // rsi
  __int64 v16; // rbx
  _QWORD *v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // rbx
  char *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  char *i; // r12
  __int64 v25; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  _QWORD *v29; // [rsp+0h] [rbp-60h]
  _QWORD *v30; // [rsp+8h] [rbp-58h]
  _QWORD *v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  char *v33; // [rsp+18h] [rbp-48h]
  char *v34; // [rsp+18h] [rbp-48h]
  char *v35; // [rsp+18h] [rbp-48h]
  char *v36; // [rsp+20h] [rbp-40h]
  _QWORD *v37; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = (v4 - *a1) >> 5;
  if ( v6 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = a2;
  v8 = a2;
  v9 = v6 == 0;
  v11 = (a1[1] - *a1) >> 5;
  v12 = 1;
  if ( !v9 )
    v12 = v11;
  v13 = __CFADD__(v11, v12);
  v14 = v11 + v12;
  v15 = a2 - v5;
  if ( v13 )
  {
    v27 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v14 )
    {
      v32 = 0;
      v16 = 32;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v14 > 0x3FFFFFFFFFFFFFFLL )
      v14 = 0x3FFFFFFFFFFFFFFLL;
    v27 = 32 * v14;
  }
  v29 = a4;
  v31 = a3;
  v35 = v7;
  v28 = sub_22077B0(v27);
  v7 = v35;
  a3 = v31;
  a4 = v29;
  v37 = (_QWORD *)v28;
  v32 = v28 + v27;
  v16 = v28 + 32;
LABEL_7:
  v17 = (_QWORD *)((char *)v37 + v15);
  if ( v17 )
  {
    v18 = *a3;
    *v17 = 6;
    v17[1] = 0;
    v17[2] = v18;
    if ( v18 != 0 && v18 != -8 && v18 != -16 )
    {
      v30 = a4;
      v33 = v7;
      sub_164C220(v17);
      a4 = v30;
      v7 = v33;
    }
    v17[3] = *a4;
  }
  if ( v7 != v5 )
  {
    v19 = v37;
    v20 = v5;
    while ( 1 )
    {
      if ( v19 )
      {
        *v19 = 6;
        v19[1] = 0;
        v21 = *((_QWORD *)v20 + 2);
        v19[2] = v21;
        if ( v21 != 0 && v21 != -8 && v21 != -16 )
        {
          v34 = v7;
          v36 = v20;
          sub_1649AC0(v19, *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v34;
          v20 = v36;
        }
        v19[3] = *((_QWORD *)v20 + 3);
      }
      v20 += 32;
      if ( v7 == v20 )
        break;
      v19 += 4;
    }
    v16 = (__int64)(v19 + 8);
  }
  if ( v7 != v4 )
  {
    do
    {
      v22 = *((_QWORD *)v8 + 2);
      *(_QWORD *)v16 = 6;
      *(_QWORD *)(v16 + 8) = 0;
      *(_QWORD *)(v16 + 16) = v22;
      if ( v22 != -8 && v22 != 0 && v22 != -16 )
        sub_1649AC0(v16, *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL);
      v23 = *((_QWORD *)v8 + 3);
      v8 += 32;
      v16 += 32;
      *(_QWORD *)(v16 - 8) = v23;
    }
    while ( v4 != v8 );
  }
  for ( i = v5; i != v4; i += 32 )
  {
    v25 = *((_QWORD *)i + 2);
    if ( v25 != -8 && v25 != 0 && v25 != -16 )
      sub_1649B30(i);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  a1[1] = (char *)v16;
  *a1 = (char *)v37;
  a1[2] = (char *)v32;
  return v32;
}
