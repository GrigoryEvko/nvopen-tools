// Function: sub_A7B050
// Address: 0xa7b050
//
unsigned __int64 __fastcall sub_A7B050(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r14d
  _QWORD *v5; // r13
  unsigned int v6; // eax
  unsigned __int64 result; // rax
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // rdx
  unsigned __int64 *i; // rdx
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 *v14; // rsi
  __int64 j; // [rsp+28h] [rbp-158h]
  _QWORD *v16; // [rsp+38h] [rbp-148h]
  unsigned __int64 v17; // [rsp+38h] [rbp-148h]
  unsigned __int64 *v18; // [rsp+40h] [rbp-140h] BYREF
  __int64 v19; // [rsp+48h] [rbp-138h]
  _BYTE v20[64]; // [rsp+50h] [rbp-130h] BYREF
  __int64 *v21; // [rsp+90h] [rbp-F0h] BYREF
  _BYTE *v22; // [rsp+98h] [rbp-E8h]
  __int64 v23; // [rsp+A0h] [rbp-E0h]
  _BYTE v24[72]; // [rsp+A8h] [rbp-D8h] BYREF
  char v25[8]; // [rsp+F0h] [rbp-90h] BYREF
  char *v26; // [rsp+F8h] [rbp-88h]
  char v27; // [rsp+108h] [rbp-78h] BYREF

  if ( !a3 )
    return 0;
  if ( a3 == 1 )
    return *a2;
  v3 = (__int64)a2;
  v4 = 0;
  v16 = &a2[a3];
  if ( a2 == v16 )
    return 0;
  v5 = &a2[a3];
  do
  {
    v6 = sub_A74480(v3);
    if ( v4 < v6 )
      v4 = v6;
    v3 += 8;
  }
  while ( v5 != (_QWORD *)v3 );
  if ( !v4 )
    return 0;
  v8 = (unsigned __int64 *)v20;
  v9 = (unsigned __int64 *)v20;
  v18 = (unsigned __int64 *)v20;
  v19 = 0x800000000LL;
  if ( v4 > 8 )
  {
    sub_C8D5F0(&v18, v20, v4, 8);
    v9 = v18;
    v8 = &v18[(unsigned int)v19];
  }
  for ( i = &v9[v4]; i != v8; ++v8 )
  {
    if ( v8 )
      *v8 = 0;
  }
  LODWORD(v19) = v4;
  for ( j = 0; j != v4; ++j )
  {
    v21 = a1;
    v11 = a2;
    v22 = v24;
    v23 = 0x800000000LL;
    do
    {
      v12 = sub_A74490(v11, (int)j - 1);
      sub_A74940((__int64)v25, (__int64)a1, v12);
      sub_A776F0((__int64)&v21, (__int64)v25);
      if ( v26 != &v27 )
        _libc_free(v26, v25);
      ++v11;
    }
    while ( v16 != v11 );
    v13 = sub_A7A280(a1, (__int64)&v21);
    v18[j] = v13;
    if ( v22 != v24 )
      _libc_free(v22, &v21);
  }
  v14 = v18;
  result = sub_A77EC0(a1, v18, (unsigned int)v19);
  if ( v18 != (unsigned __int64 *)v20 )
  {
    v17 = result;
    _libc_free(v18, v14);
    return v17;
  }
  return result;
}
