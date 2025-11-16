// Function: sub_12BC360
// Address: 0x12bc360
//
__int64 __fastcall sub_12BC360(__int64 *a1, char *a2, _QWORD *a3)
{
  char *v4; // rbx
  char *v5; // r13
  __int64 v6; // rax
  bool v8; // zf
  __int64 v10; // rsi
  __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rsi
  _QWORD *v20; // rcx
  char *v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  char *i; // r15
  __int64 v30; // rdi
  __int64 v32; // rdx
  _QWORD *v33; // [rsp+0h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v4 = (char *)a1[1];
  v5 = (char *)*a1;
  v6 = (__int64)&v4[-*a1] >> 5;
  if ( v6 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = v6 == 0;
  v10 = (v4 - v5) >> 5;
  v11 = 1;
  if ( !v8 )
    v11 = (v4 - v5) >> 5;
  v12 = __CFADD__(v10, v11);
  v13 = v10 + v11;
  v14 = (__int64 *)(a2 - v5);
  if ( v12 )
  {
    v32 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v13 )
    {
      v34 = 0;
      v15 = 32;
      v36 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x3FFFFFFFFFFFFFFLL )
      v13 = 0x3FFFFFFFFFFFFFFLL;
    v32 = 32 * v13;
  }
  v33 = a3;
  v35 = v32;
  v14 = (__int64 *)(a2 - v5);
  v36 = sub_22077B0(v32);
  a3 = v33;
  v15 = v36 + 32;
  v34 = v36 + v35;
LABEL_7:
  v16 = (__int64 *)((char *)v14 + v36);
  if ( (__int64 *)((char *)v14 + v36) )
  {
    v17 = *a3;
    *a3 = 0;
    *v16 = v17;
    v18 = a3[1];
    a3[1] = 0;
    v16[1] = v18;
    v19 = a3[2];
    a3[2] = 0;
    v16[2] = v19;
    v14 = (__int64 *)a3[3];
    a3[3] = 0;
    v16[3] = v14;
  }
  if ( a2 != v5 )
  {
    v20 = (_QWORD *)v36;
    v21 = v5;
    v22 = (_QWORD *)(v36 + a2 - v5);
    do
    {
      if ( v20 )
      {
        *v20 = *(_QWORD *)v21;
        v20[1] = *((_QWORD *)v21 + 1);
        v23 = *((_QWORD *)v21 + 2);
        *(_QWORD *)v21 = 0;
        *((_QWORD *)v21 + 1) = 0;
        v20[2] = v23;
        v14 = (__int64 *)*((_QWORD *)v21 + 3);
        v20[3] = v14;
        *((_QWORD *)v21 + 2) = 0;
        *((_QWORD *)v21 + 3) = 0;
      }
      v20 += 4;
      v21 += 32;
    }
    while ( v22 != v20 );
    v15 = (__int64)(v22 + 4);
  }
  if ( a2 == v4 )
  {
    v24 = v15;
  }
  else
  {
    v14 = (__int64 *)a2;
    v24 = v15 + v4 - a2;
    do
    {
      v25 = *v14;
      v15 += 32;
      *v14 = 0;
      v14 += 4;
      *(_QWORD *)(v15 - 32) = v25;
      v26 = *(v14 - 3);
      *(v14 - 3) = 0;
      *(_QWORD *)(v15 - 24) = v26;
      v27 = *(v14 - 2);
      *(v14 - 2) = 0;
      *(_QWORD *)(v15 - 16) = v27;
      v28 = *(v14 - 1);
      *(v14 - 1) = 0;
      *(_QWORD *)(v15 - 8) = v28;
    }
    while ( v15 != v24 );
  }
  for ( i = v5; i != v4; i += 32 )
  {
    v30 = *((_QWORD *)i + 2);
    if ( v30 )
      _libc_free(v30, v14);
    if ( *(_QWORD *)i )
      _libc_free(*(_QWORD *)i, v14);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - (_QWORD)v5);
  a1[1] = v24;
  *a1 = v36;
  a1[2] = v34;
  return v34;
}
