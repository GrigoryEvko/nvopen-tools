// Function: sub_36FEBA0
// Address: 0x36feba0
//
unsigned __int64 *__fastcall sub_36FEBA0(unsigned __int64 **a1, unsigned __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rsi
  unsigned __int64 *v5; // r12
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  bool v10; // cf
  unsigned __int64 v11; // rbx
  signed __int64 v12; // r8
  char **v13; // rbx
  __int64 v14; // rax
  char *v15; // r15
  char *v16; // rax
  unsigned __int64 *v17; // r15
  __int64 i; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  unsigned __int64 *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 *result; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // [rsp+8h] [rbp-58h]
  __int64 n; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v4 = 0x555555555555555LL;
  v5 = a1[1];
  v33 = *a1;
  v6 = (char *)v5 - (char *)*a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  v10 = __CFADD__(v7, v8);
  v11 = v7 + v8;
  v32 = v11;
  v12 = (char *)a2 - (char *)v33;
  if ( v10 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
    v32 = 0x555555555555555LL;
  }
  else
  {
    if ( !v11 )
    {
      v35 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x555555555555555LL )
      v4 = v11;
    v32 = v4;
    v28 = 24 * v4;
  }
  v30 = a3;
  v29 = sub_22077B0(v28);
  v12 = (char *)a2 - (char *)v33;
  a3 = v30;
  v35 = v29;
LABEL_7:
  v13 = (char **)(v35 + v12);
  if ( v35 + v12 )
  {
    v14 = *a3;
    if ( *a3 > 0xFFFFFFFFFFFFFFFuLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    *v13 = 0;
    v15 = 0;
    v13[1] = 0;
    v13[2] = 0;
    if ( v14 )
    {
      n = 8 * v14;
      v16 = (char *)sub_22077B0(8 * v14);
      *v13 = v16;
      v15 = &v16[n];
      v13[2] = &v16[n];
      if ( v16 != &v16[n] )
        memset(v16, 0, n);
    }
    v13[1] = v15;
  }
  v17 = v33;
  for ( i = v35; v17 != a2; i = 24 )
  {
    while ( 1 )
    {
      v20 = v17[2];
      v21 = *v17;
      if ( !i )
        break;
      *(_QWORD *)i = v21;
      v19 = v17[1];
      *(_QWORD *)(i + 16) = v20;
      *(_QWORD *)(i + 8) = v19;
      v17[2] = 0;
      *v17 = 0;
LABEL_16:
      v17 += 3;
      i += 24;
      if ( v17 == a2 )
        goto LABEL_20;
    }
    if ( !v21 )
      goto LABEL_16;
    j_j___libc_free_0(v21);
    v17 += 3;
  }
LABEL_20:
  v22 = i + 24;
  if ( a2 != v5 )
  {
    v23 = a2;
    v24 = i + 24;
    do
    {
      v25 = *v23;
      v23 += 3;
      v24 += 24;
      *(_QWORD *)(v24 - 24) = v25;
      *(_QWORD *)(v24 - 16) = *(v23 - 2);
      *(_QWORD *)(v24 - 8) = *(v23 - 1);
    }
    while ( v23 != v5 );
    v22 += 8 * ((unsigned __int64)((char *)v23 - (char *)a2 - 24) >> 3) + 24;
  }
  v26 = (unsigned __int64)v33;
  if ( v33 )
  {
    v34 = v22;
    j_j___libc_free_0(v26);
    v22 = v34;
  }
  a1[1] = (unsigned __int64 *)v22;
  *a1 = (unsigned __int64 *)v35;
  result = (unsigned __int64 *)(v35 + 24 * v32);
  a1[2] = result;
  return result;
}
