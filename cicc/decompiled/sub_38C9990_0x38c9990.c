// Function: sub_38C9990
// Address: 0x38c9990
//
unsigned __int64 *__fastcall sub_38C9990(unsigned __int64 *a1, char *a2, _QWORD *a3)
{
  char *v3; // r12
  __int64 v4; // rax
  char *v6; // r14
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rsi
  __int64 v13; // rbx
  char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r13
  char *v19; // rbx
  __int64 v20; // rsi
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // [rsp+0h] [rbp-60h]
  unsigned __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned __int64 v30; // [rsp+20h] [rbp-40h]
  unsigned __int64 v31; // [rsp+28h] [rbp-38h]

  v3 = (char *)a1[1];
  v30 = *a1;
  v4 = (__int64)&v3[-*a1] >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = a2;
  v7 = (__int64)&v3[-v30] >> 5;
  v8 = v4 == 0;
  v9 = 1;
  if ( !v8 )
    v9 = (__int64)&v3[-v30] >> 5;
  v10 = __CFADD__(v7, v9);
  v11 = v7 + v9;
  v12 = &a2[-v30];
  if ( v10 )
  {
    v25 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v28 = 0;
      v13 = 32;
      v31 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x3FFFFFFFFFFFFFFLL )
      v11 = 0x3FFFFFFFFFFFFFFLL;
    v25 = 32 * v11;
  }
  v27 = a3;
  v26 = sub_22077B0(v25);
  a3 = v27;
  v31 = v26;
  v28 = v26 + v25;
  v13 = v26 + 32;
LABEL_7:
  v14 = &v12[v31];
  if ( v14 )
  {
    *(_QWORD *)v14 = *a3;
    v15 = a3[1];
    a3[1] = 0;
    *((_QWORD *)v14 + 1) = v15;
    v16 = a3[2];
    a3[2] = 0;
    *((_QWORD *)v14 + 2) = v16;
    v17 = a3[3];
    a3[3] = 0;
    *((_QWORD *)v14 + 3) = v17;
  }
  if ( a2 != (char *)v30 )
  {
    v18 = (_QWORD *)v31;
    v19 = (char *)v30;
    while ( !v18 )
    {
      v21 = *((_QWORD *)v19 + 1);
      if ( !v21 )
        goto LABEL_12;
      j_j___libc_free_0(v21);
      v19 += 32;
      v20 = 32;
      if ( v19 == a2 )
      {
LABEL_17:
        v13 = (__int64)(v18 + 8);
        goto LABEL_18;
      }
LABEL_13:
      v18 = (_QWORD *)v20;
    }
    *v18 = *(_QWORD *)v19;
    v18[1] = *((_QWORD *)v19 + 1);
    v18[2] = *((_QWORD *)v19 + 2);
    v18[3] = *((_QWORD *)v19 + 3);
    *((_QWORD *)v19 + 3) = 0;
    *((_QWORD *)v19 + 1) = 0;
LABEL_12:
    v19 += 32;
    v20 = (__int64)(v18 + 4);
    if ( v19 == a2 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v3 )
  {
    v22 = v13;
    do
    {
      v23 = *(_QWORD *)v6;
      v6 += 32;
      v22 += 32;
      *(_QWORD *)(v22 - 32) = v23;
      *(_QWORD *)(v22 - 24) = *((_QWORD *)v6 - 3);
      *(_QWORD *)(v22 - 16) = *((_QWORD *)v6 - 2);
      *(_QWORD *)(v22 - 8) = *((_QWORD *)v6 - 1);
    }
    while ( v6 != v3 );
    v13 += v3 - a2;
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  *a1 = v31;
  a1[1] = v13;
  a1[2] = v28;
  return a1;
}
