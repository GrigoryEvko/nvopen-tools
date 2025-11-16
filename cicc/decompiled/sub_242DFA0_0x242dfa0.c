// Function: sub_242DFA0
// Address: 0x242dfa0
//
unsigned __int64 *__fastcall sub_242DFA0(unsigned __int64 *a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  __int64 v8; // rdi
  bool v9; // zf
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 *v16; // rbx
  __int64 *v17; // rsi
  __int64 v18; // rdi
  _QWORD *v19; // r15
  _QWORD *v20; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (__int64)(v3 - *a1) >> 4;
  if ( v5 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = (__int64)(v3 - v4) >> 4;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = (__int64)(v3 - v4) >> 4;
  v11 = __CFADD__(v8, v10);
  v12 = v8 + v10;
  v13 = (char *)a2 - v4;
  if ( v11 )
  {
    v22 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v12 )
    {
      v24 = 0;
      v14 = 16;
      v26 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x7FFFFFFFFFFFFFFLL )
      v12 = 0x7FFFFFFFFFFFFFFLL;
    v22 = 16 * v12;
  }
  v23 = sub_22077B0(v22);
  v13 = (char *)a2 - v4;
  v26 = v23;
  v24 = v23 + v22;
  v14 = v23 + 16;
LABEL_7:
  if ( &v13[v26] )
    sub_C88FD0((__int64)&v13[v26], a3);
  if ( a2 != (__int64 *)v4 )
  {
    v15 = v26;
    v16 = (__int64 *)v4;
    while ( 1 )
    {
      if ( v15 )
        sub_C88FD0(v15, v16);
      v16 += 2;
      if ( a2 == v16 )
        break;
      v15 += 16;
    }
    v14 = v15 + 32;
  }
  while ( (__int64 *)v3 != a2 )
  {
    v17 = a2;
    v18 = v14;
    a2 += 2;
    v14 += 16;
    sub_C88FD0(v18, v17);
  }
  v19 = (_QWORD *)v4;
  if ( v4 != v3 )
  {
    do
    {
      v20 = v19;
      v19 += 2;
      sub_C88FF0(v20);
    }
    while ( v19 != (_QWORD *)v3 );
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v26;
  a1[1] = v14;
  a1[2] = v24;
  return a1;
}
