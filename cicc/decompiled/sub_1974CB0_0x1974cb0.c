// Function: sub_1974CB0
// Address: 0x1974cb0
//
__int64 __fastcall sub_1974CB0(__int64 a1, char *a2, _QWORD *a3)
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
  _QWORD *v13; // r8
  __int64 v14; // rax
  __int64 v15; // rbx
  char *v16; // rax
  _QWORD *v17; // r8
  char *v18; // rdi
  _BYTE *v19; // r9
  _BYTE *v20; // rdx
  size_t v21; // rdx
  char *v22; // rax
  char *v23; // r15
  __int64 i; // rbx
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdx
  char *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  char *v33; // rdi
  __int64 result; // rax
  __int64 v35; // rax
  const void **v36; // [rsp+8h] [rbp-58h]
  _QWORD *v37; // [rsp+8h] [rbp-58h]
  _QWORD *v38; // [rsp+8h] [rbp-58h]
  _QWORD *v39; // [rsp+10h] [rbp-50h]
  size_t v40; // [rsp+10h] [rbp-50h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  char *v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v4 = 0x555555555555555LL;
  v5 = *(char **)(a1 + 8);
  v42 = *(char **)a1;
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
  v41 = v11;
  v12 = a2 - v42;
  if ( v10 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v41 = 0x555555555555555LL;
  }
  else
  {
    if ( !v11 )
    {
      v44 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x555555555555555LL )
      v4 = v11;
    v41 = v4;
    a1 = 24 * v4;
  }
  v38 = a3;
  v35 = sub_22077B0(a1);
  v12 = a2 - v42;
  a3 = v38;
  v44 = v35;
LABEL_7:
  v13 = (_QWORD *)(v44 + v12);
  if ( v13 )
  {
    v14 = a3[1] - *a3;
    *v13 = 0;
    v13[1] = 0;
    v15 = v14;
    v13[2] = 0;
    if ( v14 )
    {
      if ( v14 < 0 )
        sub_4261EA(a1, v4, a3);
      v36 = (const void **)a3;
      v39 = v13;
      v16 = (char *)sub_22077B0(v14);
      v17 = v39;
      v18 = v16;
      v19 = *v36;
      v20 = v36[1];
      *v39 = v16;
      v39[1] = v16;
      v39[2] = &v16[v15];
      v21 = v20 - v19;
      if ( v21 )
      {
        v37 = v39;
        v40 = v21;
        v22 = (char *)memmove(v16, v19, v21);
        v17 = v37;
        v21 = v40;
        v18 = v22;
      }
      v17[1] = &v18[v21];
    }
    else
    {
      v13[1] = 0;
    }
  }
  v23 = v42;
  for ( i = v44; v23 != a2; i = 24 )
  {
    while ( 1 )
    {
      v26 = *((_QWORD *)v23 + 2);
      v27 = *(_QWORD *)v23;
      if ( !i )
        break;
      *(_QWORD *)i = v27;
      v25 = *((_QWORD *)v23 + 1);
      *(_QWORD *)(i + 16) = v26;
      *(_QWORD *)(i + 8) = v25;
      *((_QWORD *)v23 + 2) = 0;
      *(_QWORD *)v23 = 0;
LABEL_16:
      v23 += 24;
      i += 24;
      if ( v23 == a2 )
        goto LABEL_20;
    }
    v28 = v26 - v27;
    if ( !v27 )
      goto LABEL_16;
    j_j___libc_free_0(v27, v28);
    v23 += 24;
  }
LABEL_20:
  v29 = i + 24;
  if ( a2 != v5 )
  {
    v30 = a2;
    v31 = i + 24;
    do
    {
      v32 = *(_QWORD *)v30;
      v30 += 24;
      v31 += 24;
      *(_QWORD *)(v31 - 24) = v32;
      *(_QWORD *)(v31 - 16) = *((_QWORD *)v30 - 2);
      *(_QWORD *)(v31 - 8) = *((_QWORD *)v30 - 1);
    }
    while ( v30 != v5 );
    v29 += 8 * ((unsigned __int64)(v30 - a2 - 24) >> 3) + 24;
  }
  v33 = v42;
  if ( v42 )
  {
    v43 = v29;
    j_j___libc_free_0(v33, v9[2] - (_QWORD)v33);
    v29 = v43;
  }
  v9[1] = v29;
  *v9 = v44;
  result = v44 + 24 * v41;
  v9[2] = result;
  return result;
}
