// Function: sub_1563520
// Address: 0x1563520
//
__int64 __fastcall sub_1563520(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  unsigned int v4; // r14d
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 *v7; // rax
  __int64 *v8; // rdx
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 i; // [rsp+30h] [rbp-160h]
  __int64 *v14; // [rsp+38h] [rbp-158h]
  __int64 v15; // [rsp+38h] [rbp-158h]
  __int64 v16; // [rsp+48h] [rbp-148h] BYREF
  __int64 *v17; // [rsp+50h] [rbp-140h] BYREF
  __int64 v18; // [rsp+58h] [rbp-138h]
  _BYTE v19[64]; // [rsp+60h] [rbp-130h] BYREF
  _QWORD v20[2]; // [rsp+A0h] [rbp-F0h] BYREF
  int v21; // [rsp+B0h] [rbp-E0h] BYREF
  _QWORD *v22; // [rsp+B8h] [rbp-D8h]
  int *v23; // [rsp+C0h] [rbp-D0h]
  int *v24; // [rsp+C8h] [rbp-C8h]
  __int64 v25; // [rsp+D0h] [rbp-C0h]
  __int64 v26; // [rsp+D8h] [rbp-B8h]
  __int64 v27; // [rsp+E0h] [rbp-B0h]
  __int64 v28; // [rsp+E8h] [rbp-A8h]
  __int64 v29; // [rsp+F0h] [rbp-A0h]
  __int64 v30; // [rsp+F8h] [rbp-98h]
  __m128i v31; // [rsp+100h] [rbp-90h] BYREF
  _QWORD *v32; // [rsp+118h] [rbp-78h]

  if ( !a3 )
    return 0;
  if ( a3 == 1 )
    return *a2;
  v14 = &a2[a3];
  if ( v14 == a2 )
    return 0;
  v3 = a2;
  v4 = 0;
  do
  {
    v31.m128i_i64[0] = *v3;
    v5 = sub_15601D0((__int64)&v31);
    if ( v4 < v5 )
      v4 = v5;
    ++v3;
  }
  while ( v14 != v3 );
  if ( !v4 )
    return 0;
  v7 = (__int64 *)v19;
  v17 = (__int64 *)v19;
  v18 = 0x800000000LL;
  if ( v4 > 8 )
  {
    sub_16CD150(&v17, v19, v4, 8);
    v7 = v17;
  }
  LODWORD(v18) = v4;
  v8 = &v7[v4];
  do
  {
    if ( v7 )
      *v7 = 0;
    ++v7;
  }
  while ( v8 != v7 );
  for ( i = 0; i != v4; ++i )
  {
    v20[0] = 0;
    v21 = 0;
    v9 = a2;
    v23 = &v21;
    v24 = &v21;
    v22 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    do
    {
      v10 = *v9++;
      v16 = v10;
      v11 = sub_15601E0(&v16, (int)i - 1);
      sub_1563030(&v31, v11);
      sub_15625F0(v20, &v31);
      sub_155CC10(v32);
    }
    while ( v14 != v9 );
    v12 = sub_1560BF0(a1, v20);
    v17[i] = v12;
    sub_155CC10(v22);
  }
  result = sub_155F990(a1, v17, (unsigned int)v18);
  if ( v17 != (__int64 *)v19 )
  {
    v15 = result;
    _libc_free((unsigned __int64)v17);
    return v15;
  }
  return result;
}
