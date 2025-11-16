// Function: sub_ED3260
// Address: 0xed3260
//
__int64 __fastcall sub_ED3260(__int64 a1, char *a2, __int64 a3)
{
  __int64 v4; // rsi
  char *v5; // r12
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r13
  bool v10; // cf
  unsigned __int64 v11; // rbx
  signed __int64 v12; // r8
  _QWORD *v13; // rbx
  _BYTE *v14; // rax
  _BYTE *v15; // rsi
  unsigned __int64 v16; // r9
  __int64 v17; // rax
  char *v18; // rdi
  size_t v19; // r15
  char *v20; // r15
  __int64 i; // rbx
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rdx
  char *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdi
  char *v30; // rdi
  __int64 result; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  char *v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v4 = 0x555555555555555LL;
  v5 = *(char **)(a1 + 8);
  v37 = *(char **)a1;
  v6 = (__int64)&v5[-*(_QWORD *)a1];
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (_QWORD *)a1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  v10 = __CFADD__(v7, v8);
  v11 = v7 + v8;
  v36 = v11;
  v12 = a2 - v37;
  if ( v10 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v36 = 0x555555555555555LL;
  }
  else
  {
    if ( !v11 )
    {
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x555555555555555LL )
      v4 = v11;
    v36 = v4;
    a1 = 24 * v4;
  }
  v34 = a3;
  v32 = sub_22077B0(a1);
  v12 = a2 - v37;
  a3 = v34;
  v39 = v32;
LABEL_7:
  v13 = (_QWORD *)(v39 + v12);
  if ( v39 + v12 )
  {
    v14 = *(_BYTE **)(a3 + 8);
    v15 = *(_BYTE **)a3;
    *v13 = 0;
    v13[1] = 0;
    v13[2] = 0;
    v16 = v14 - v15;
    if ( v14 == v15 )
    {
      v19 = 0;
      v18 = 0;
    }
    else
    {
      if ( v16 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(a1, v15, a3);
      v33 = a3;
      v35 = v14 - v15;
      v17 = sub_22077B0(v16);
      v16 = v35;
      v18 = (char *)v17;
      v14 = *(_BYTE **)(v33 + 8);
      v15 = *(_BYTE **)v33;
      v19 = (size_t)&v14[-*(_QWORD *)v33];
    }
    *v13 = v18;
    v13[1] = v18;
    v13[2] = &v18[v16];
    if ( v14 != v15 )
      v18 = (char *)memmove(v18, v15, v19);
    v13[1] = &v18[v19];
  }
  v20 = v37;
  for ( i = v39; v20 != a2; i = 24 )
  {
    while ( 1 )
    {
      v23 = *((_QWORD *)v20 + 2);
      v24 = *(_QWORD *)v20;
      if ( !i )
        break;
      *(_QWORD *)i = v24;
      v22 = *((_QWORD *)v20 + 1);
      *(_QWORD *)(i + 16) = v23;
      *(_QWORD *)(i + 8) = v22;
      *((_QWORD *)v20 + 2) = 0;
      *(_QWORD *)v20 = 0;
LABEL_17:
      v20 += 24;
      i += 24;
      if ( v20 == a2 )
        goto LABEL_21;
    }
    v25 = v23 - v24;
    if ( !v24 )
      goto LABEL_17;
    j_j___libc_free_0(v24, v25);
    v20 += 24;
  }
LABEL_21:
  v26 = i + 24;
  if ( a2 != v5 )
  {
    v27 = a2;
    v28 = i + 24;
    do
    {
      v29 = *(_QWORD *)v27;
      v27 += 24;
      v28 += 24;
      *(_QWORD *)(v28 - 24) = v29;
      *(_QWORD *)(v28 - 16) = *((_QWORD *)v27 - 2);
      *(_QWORD *)(v28 - 8) = *((_QWORD *)v27 - 1);
    }
    while ( v27 != v5 );
    v26 += 8 * ((unsigned __int64)(v27 - a2 - 24) >> 3) + 24;
  }
  v30 = v37;
  if ( v37 )
  {
    v38 = v26;
    j_j___libc_free_0(v30, v9[2] - (_QWORD)v30);
    v26 = v38;
  }
  v9[1] = v26;
  *v9 = v39;
  result = v39 + 24 * v36;
  v9[2] = result;
  return result;
}
