// Function: sub_C6D0A0
// Address: 0xc6d0a0
//
__int16 **__fastcall sub_C6D0A0(__int16 **a1, __int16 *a2, __int64 a3)
{
  __int16 *v3; // r15
  __int16 *v4; // r12
  __int16 *v5; // r13
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rcx
  __int64 v11; // rbx
  _WORD *v12; // rcx
  __int64 v13; // r14
  __int16 *v14; // rbx
  __int16 *v15; // rsi
  __int64 v16; // rdi
  unsigned __int16 *v17; // r15
  unsigned __int16 *v18; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v3 = a2;
  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v4 - (char *)*a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v4 - (char *)v5) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * (((char *)v4 - (char *)v5) >> 3);
  v10 = (char *)a2 - (char *)v5;
  if ( v8 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v22 = 0;
      v11 = 40;
      v24 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x333333333333333LL )
      v9 = 0x333333333333333LL;
    v20 = 40 * v9;
  }
  v21 = sub_22077B0(v20);
  v10 = (char *)a2 - (char *)v5;
  v24 = v21;
  v22 = v21 + v20;
  v11 = v21 + 40;
LABEL_7:
  v12 = (_WORD *)(v24 + v10);
  if ( v12 )
    *v12 = 0;
  if ( a2 != v5 )
  {
    v13 = v24;
    v14 = v5;
    while ( 1 )
    {
      if ( v13 )
        sub_C6CEC0(v13, v14, a3);
      v14 += 20;
      if ( a2 == v14 )
        break;
      v13 += 40;
    }
    v11 = v13 + 80;
  }
  if ( a2 != v4 )
  {
    do
    {
      v15 = v3;
      v16 = v11;
      v3 += 20;
      v11 += 40;
      sub_C6CEC0(v16, v15, a3);
    }
    while ( v4 != v3 );
  }
  v17 = (unsigned __int16 *)v5;
  if ( v5 != v4 )
  {
    do
    {
      v18 = v17;
      v17 += 20;
      sub_C6BC50(v18);
    }
    while ( v17 != (unsigned __int16 *)v4 );
  }
  if ( v5 )
    j_j___libc_free_0(v5, (char *)a1[2] - (char *)v5);
  *a1 = (__int16 *)v24;
  a1[1] = (__int16 *)v11;
  a1[2] = (__int16 *)v22;
  return a1;
}
