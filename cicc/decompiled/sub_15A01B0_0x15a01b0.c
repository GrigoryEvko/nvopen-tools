// Function: sub_15A01B0
// Address: 0x15a01b0
//
__int64 __fastcall sub_15A01B0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 **v4; // r14
  __int64 v5; // rbx
  int v6; // eax
  __m128i v7; // xmm0
  __int64 v8; // rdx
  __int64 v9; // r8
  int v10; // r9d
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v14; // r11d
  _QWORD *v15; // rcx
  __int64 v16; // r11
  _QWORD *v17; // rax
  int i; // [rsp+20h] [rbp-80h]
  int v19; // [rsp+24h] [rbp-7Ch]
  __m128i v20; // [rsp+30h] [rbp-70h] BYREF
  __int64 v21; // [rsp+40h] [rbp-60h]
  int v22; // [rsp+50h] [rbp-50h] BYREF
  __m128i v23; // [rsp+58h] [rbp-48h]
  __int64 v24; // [rsp+68h] [rbp-38h]

  result = sub_159ABB0(a1, a2);
  if ( !result )
  {
    v4 = (__int64 **)sub_16463B0(*(_QWORD *)*a1, (unsigned int)a2);
    v5 = **v4;
    v20.m128i_i64[0] = (__int64)v4;
    v20.m128i_i64[1] = (__int64)a1;
    v21 = a2;
    v22 = sub_1597240(a1, (__int64)&a1[a2]);
    v6 = sub_1597ED0(v20.m128i_i64, &v22);
    v7 = _mm_loadu_si128(&v20);
    v22 = v6;
    v24 = v21;
    v23 = v7;
    v8 = *(unsigned int *)(v5 + 1640);
    if ( !(_DWORD)v8 )
      return sub_159FE00(v5 + 1616, (__int64)v4, a1, a2, (__int64)&v22);
    v9 = *(_QWORD *)(v5 + 1624);
    v10 = v8 - 1;
    v11 = ((_DWORD)v8 - 1) & (unsigned int)v6;
    v12 = (__int64 *)(v9 + 8 * v11);
    v19 = v11;
    v13 = *v12;
    if ( *v12 == -8 )
      return sub_159FE00(v5 + 1616, (__int64)v4, a1, a2, (__int64)&v22);
    for ( i = 1; ; ++i )
    {
      if ( v13 != -16 && v23.m128i_i64[0] == *(_QWORD *)v13 )
      {
        v14 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
        if ( v21 == v14 )
        {
          if ( !v14 )
          {
LABEL_15:
            if ( v12 != (__int64 *)(v9 + 8 * v8) )
              return *v12;
            return sub_159FE00(v5 + 1616, (__int64)v4, a1, a2, (__int64)&v22);
          }
          v15 = (_QWORD *)v23.m128i_i64[1];
          v16 = v23.m128i_i64[1] + 8 + 8LL * (unsigned int)(v14 - 1);
          v17 = (_QWORD *)(-24 * v21 + v13);
          while ( *v15 == *v17 )
          {
            ++v15;
            v17 += 3;
            if ( (_QWORD *)v16 == v15 )
              goto LABEL_15;
          }
        }
      }
      v12 = (__int64 *)(v9 + 8LL * (v10 & (unsigned int)(v19 + i)));
      v19 = v10 & (v19 + i);
      v13 = *v12;
      if ( *v12 == -8 )
        return sub_159FE00(v5 + 1616, (__int64)v4, a1, a2, (__int64)&v22);
    }
  }
  return result;
}
