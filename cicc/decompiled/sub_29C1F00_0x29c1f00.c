// Function: sub_29C1F00
// Address: 0x29c1f00
//
unsigned __int64 *__fastcall sub_29C1F00(unsigned __int64 *a1, __int16 *a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  unsigned __int16 *v8; // rsi
  __int16 *v9; // r15
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int16 *v15; // rbx
  __int16 *v16; // rsi
  __int64 v17; // rdi
  unsigned __int16 *v18; // r14
  unsigned __int16 *v19; // rdi
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v3 - *a1) >> 3);
  if ( v5 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  v8 = (unsigned __int16 *)a3;
  if ( v5 )
    v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v3 - v4) >> 3);
  v9 = a2;
  v10 = __CFADD__(v6, v5);
  v11 = v6 - 0x3333333333333333LL * ((__int64)(v3 - v4) >> 3);
  v12 = (char *)a2 - v4;
  if ( v10 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v24 = 0;
      v13 = 40;
      v26 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x333333333333333LL )
      v11 = 0x333333333333333LL;
    v21 = 40 * v11;
  }
  v22 = sub_22077B0(v21);
  v12 = (char *)a2 - v4;
  v8 = (unsigned __int16 *)a3;
  v26 = v22;
  v24 = v22 + v21;
  v13 = v22 + 40;
LABEL_7:
  if ( &v12[v26] )
    sub_C6A4F0((__int64)&v12[v26], v8);
  if ( a2 != (__int16 *)v4 )
  {
    v14 = v26;
    v15 = (__int16 *)v4;
    while ( 1 )
    {
      if ( v14 )
      {
        v27 = v14;
        sub_C6CEC0(v14, v15, a3);
        v14 = v27;
      }
      v15 += 20;
      if ( a2 == v15 )
        break;
      v14 += 40;
    }
    v13 = v14 + 80;
  }
  if ( a2 != (__int16 *)v3 )
  {
    do
    {
      v16 = v9;
      v17 = v13;
      v9 += 20;
      v13 += 40;
      sub_C6CEC0(v17, v16, a3);
    }
    while ( (__int16 *)v3 != v9 );
  }
  v18 = (unsigned __int16 *)v4;
  if ( v4 != v3 )
  {
    do
    {
      v19 = v18;
      v18 += 20;
      sub_C6BC50(v19);
    }
    while ( v18 != (unsigned __int16 *)v3 );
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v26;
  a1[1] = v13;
  a1[2] = v24;
  return a1;
}
