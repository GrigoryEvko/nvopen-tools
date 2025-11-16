// Function: sub_159F090
// Address: 0x159f090
//
__int64 __fastcall sub_159F090(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  bool v5; // r15
  bool v6; // al
  __int64 v7; // rdx
  __int64 v8; // rcx
  bool v9; // r12
  __int64 v10; // rbx
  int v11; // eax
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // r9
  unsigned int v15; // eax
  __int64 *v16; // r8
  __int64 v17; // rcx
  __int64 *v18; // rbx
  int v20; // r11d
  _QWORD *v21; // rdi
  _QWORD *v22; // rcx
  __int64 v23; // r11
  int i; // [rsp+4h] [rbp-9Ch]
  __m128i v27; // [rsp+30h] [rbp-70h] BYREF
  __int64 v28; // [rsp+40h] [rbp-60h]
  int v29; // [rsp+50h] [rbp-50h] BYREF
  __m128i v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+68h] [rbp-38h]

  if ( !a3 )
    return sub_1598F00(a1);
  v4 = a2;
  v5 = *(_BYTE *)(*a2 + 16) == 9;
  v6 = sub_1593BB0(*a2, (__int64)a2, a3, a4);
  v9 = v6;
  if ( v5 || v6 )
  {
    if ( (_DWORD)a3 )
    {
      v18 = a2;
      do
      {
        if ( !sub_1593BB0(*v18, (__int64)a2, v7, v8) )
          v9 = 0;
        if ( *(_BYTE *)(*v18 + 16) != 9 )
          v5 = 0;
        ++v18;
      }
      while ( &a2[(unsigned int)(a3 - 1) + 1] != v18 );
      v4 = a2;
    }
    if ( !v9 )
    {
      if ( v5 )
        return sub_1599EF0(a1);
      goto LABEL_4;
    }
    return sub_1598F00(a1);
  }
LABEL_4:
  v10 = **a1;
  v27.m128i_i64[0] = (__int64)a1;
  v27.m128i_i64[1] = (__int64)v4;
  v28 = a3;
  v29 = sub_1597240(v4, (__int64)&v4[a3]);
  v11 = sub_1597BF0(v27.m128i_i64, &v29);
  v12 = _mm_loadu_si128(&v27);
  v29 = v11;
  v31 = v28;
  v30 = v12;
  v13 = *(unsigned int *)(v10 + 1608);
  if ( !(_DWORD)v13 )
    return sub_159ECE0(v10 + 1584, (__int64)a1, v4, a3, (__int64)&v29);
  v14 = *(_QWORD *)(v10 + 1592);
  v15 = (v13 - 1) & v11;
  v16 = (__int64 *)(v14 + 8LL * v15);
  v17 = *v16;
  if ( *v16 == -8 )
    return sub_159ECE0(v10 + 1584, (__int64)a1, v4, a3, (__int64)&v29);
  for ( i = 1; ; ++i )
  {
    if ( v17 == -16 )
      goto LABEL_9;
    if ( v30.m128i_i64[0] != *(_QWORD *)v17 )
      goto LABEL_9;
    v20 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
    if ( v28 != v20 )
      goto LABEL_9;
    if ( !v20 )
      break;
    v21 = (_QWORD *)v30.m128i_i64[1];
    v22 = (_QWORD *)(-24 * v28 + v17);
    v23 = v30.m128i_i64[1] + 8 + 8LL * (unsigned int)(v20 - 1);
    while ( *v21 == *v22 )
    {
      ++v21;
      v22 += 3;
      if ( (_QWORD *)v23 == v21 )
        goto LABEL_30;
    }
LABEL_9:
    v15 = (v13 - 1) & (i + v15);
    v16 = (__int64 *)(v14 + 8LL * v15);
    v17 = *v16;
    if ( *v16 == -8 )
      return sub_159ECE0(v10 + 1584, (__int64)a1, v4, a3, (__int64)&v29);
  }
LABEL_30:
  if ( v16 == (__int64 *)(v14 + 8 * v13) )
    return sub_159ECE0(v10 + 1584, (__int64)a1, v4, a3, (__int64)&v29);
  return *v16;
}
