// Function: sub_2F18600
// Address: 0x2f18600
//
unsigned __int64 *__fastcall sub_2F18600(unsigned __int64 *a1, char *a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // zf
  char *v7; // r13
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  __int64 v12; // rbx
  char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  char *v17; // r15
  _QWORD *i; // rbx
  __int64 v19; // rax
  unsigned __int64 *v20; // r12
  unsigned __int64 *v21; // r14
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  char *v27; // [rsp+10h] [rbp-60h]
  unsigned __int64 v28; // [rsp+18h] [rbp-58h]
  unsigned __int64 v29; // [rsp+28h] [rbp-48h]
  unsigned __int64 v30; // [rsp+30h] [rbp-40h]

  v27 = (char *)a1[1];
  v4 = (__int64)&v27[-*a1] >> 5;
  v29 = *a1;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = (__int64)&v27[-*a1] >> 5;
  v6 = v4 == 0;
  v7 = a2;
  v8 = 1;
  if ( !v6 )
    v8 = (__int64)&v27[-*a1] >> 5;
  v9 = __CFADD__(v5, v8);
  v10 = v5 + v8;
  v11 = &a2[-v29];
  if ( v9 )
  {
    v25 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v28 = 0;
      v12 = 32;
      v30 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v25 = 32 * v10;
  }
  v26 = sub_22077B0(v25);
  v11 = &a2[-v29];
  v30 = v26;
  v28 = v26 + v25;
  v12 = v26 + 32;
LABEL_7:
  v13 = &v11[v30];
  if ( &v11[v30] )
  {
    *(_QWORD *)v13 = *a3;
    v14 = a3[1];
    a3[1] = 0;
    *((_QWORD *)v13 + 1) = v14;
    v15 = a3[2];
    a3[2] = 0;
    *((_QWORD *)v13 + 2) = v15;
    v16 = a3[3];
    a3[3] = 0;
    *((_QWORD *)v13 + 3) = v16;
  }
  v17 = (char *)v29;
  if ( a2 != (char *)v29 )
  {
    for ( i = (_QWORD *)v30; !i; i = (_QWORD *)v19 )
    {
      v20 = (unsigned __int64 *)*((_QWORD *)v17 + 2);
      v21 = (unsigned __int64 *)*((_QWORD *)v17 + 1);
      if ( v20 != v21 )
      {
        do
        {
          if ( (unsigned __int64 *)*v21 != v21 + 2 )
            j_j___libc_free_0(*v21);
          v21 += 7;
        }
        while ( v20 != v21 );
        v21 = (unsigned __int64 *)*((_QWORD *)v17 + 1);
      }
      if ( !v21 )
        goto LABEL_12;
      v17 += 32;
      j_j___libc_free_0((unsigned __int64)v21);
      v19 = 32;
      if ( v17 == a2 )
      {
LABEL_22:
        v12 = (__int64)(i + 8);
        goto LABEL_23;
      }
LABEL_13:
      ;
    }
    *i = *(_QWORD *)v17;
    i[1] = *((_QWORD *)v17 + 1);
    i[2] = *((_QWORD *)v17 + 2);
    i[3] = *((_QWORD *)v17 + 3);
    *((_QWORD *)v17 + 3) = 0;
    *((_QWORD *)v17 + 2) = 0;
    *((_QWORD *)v17 + 1) = 0;
LABEL_12:
    v17 += 32;
    v19 = (__int64)(i + 4);
    if ( v17 == a2 )
      goto LABEL_22;
    goto LABEL_13;
  }
LABEL_23:
  if ( a2 == v27 )
  {
    v22 = v12;
  }
  else
  {
    v22 = v12 + v27 - a2;
    do
    {
      v23 = *(_QWORD *)v7;
      v12 += 32;
      v7 += 32;
      *(_QWORD *)(v12 - 32) = v23;
      *(_QWORD *)(v12 - 24) = *((_QWORD *)v7 - 3);
      *(_QWORD *)(v12 - 16) = *((_QWORD *)v7 - 2);
      *(_QWORD *)(v12 - 8) = *((_QWORD *)v7 - 1);
    }
    while ( v12 != v22 );
  }
  if ( v29 )
    j_j___libc_free_0(v29);
  *a1 = v30;
  a1[1] = v22;
  a1[2] = v28;
  return a1;
}
