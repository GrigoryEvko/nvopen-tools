// Function: sub_2647470
// Address: 0x2647470
//
unsigned __int64 *__fastcall sub_2647470(unsigned __int64 *a1, char *a2, _QWORD *a3)
{
  char *v4; // r14
  char *v5; // r12
  unsigned __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // r9
  __int64 v12; // rbx
  char *v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // r13
  char *v16; // rbx
  __int64 v17; // rdi
  volatile signed __int32 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // [rsp+0h] [rbp-60h]
  unsigned __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h]
  unsigned __int64 v28; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v5 = (char *)a1[1];
  v6 = *a1;
  v27 = *a1;
  v7 = (__int64)&v5[-*a1] >> 4;
  if ( v7 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = (__int64)&v5[-v6] >> 4;
  v9 = __CFADD__(v8, v7);
  v10 = v8 + v7;
  v11 = &a2[-v27];
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFFF0LL;
LABEL_28:
    v24 = a3;
    v23 = sub_22077B0(v22);
    v11 = &a2[-v27];
    a3 = v24;
    v28 = v23;
    v25 = v23 + v22;
    v12 = v23 + 16;
    goto LABEL_7;
  }
  if ( v10 )
  {
    if ( v10 > 0x7FFFFFFFFFFFFFFLL )
      v10 = 0x7FFFFFFFFFFFFFFLL;
    v22 = 16 * v10;
    goto LABEL_28;
  }
  v25 = 0;
  v12 = 16;
  v28 = 0;
LABEL_7:
  v13 = &v11[v28];
  if ( v13 )
  {
    *(_QWORD *)v13 = *a3;
    v14 = a3[1];
    *((_QWORD *)v13 + 1) = v14;
    if ( v14 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd((volatile signed __int32 *)(v14 + 8), 1u);
      else
        ++*(_DWORD *)(v14 + 8);
    }
  }
  if ( a2 != (char *)v27 )
  {
    v15 = (_QWORD *)v28;
    v16 = (char *)v27;
    while ( 1 )
    {
      if ( v15 )
      {
        *v15 = *(_QWORD *)v16;
        v17 = *((_QWORD *)v16 + 1);
        *((_QWORD *)v16 + 1) = 0;
        v15[1] = v17;
        *(_QWORD *)v16 = 0;
      }
      v18 = (volatile signed __int32 *)*((_QWORD *)v16 + 1);
      if ( v18 )
        sub_A191D0(v18);
      v16 += 16;
      if ( v16 == a2 )
        break;
      v15 += 2;
    }
    v12 = (__int64)(v15 + 4);
  }
  if ( a2 != v5 )
  {
    v19 = v12;
    do
    {
      v20 = *(_QWORD *)v4;
      v4 += 16;
      v19 += 16;
      *(_QWORD *)(v19 - 16) = v20;
      *(_QWORD *)(v19 - 8) = *((_QWORD *)v4 - 1);
    }
    while ( v4 != v5 );
    v12 += v5 - a2;
  }
  if ( v27 )
    j_j___libc_free_0(v27);
  *a1 = v28;
  a1[1] = v12;
  a1[2] = v25;
  return a1;
}
