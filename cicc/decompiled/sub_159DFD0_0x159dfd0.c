// Function: sub_159DFD0
// Address: 0x159dfd0
//
__int64 __fastcall sub_159DFD0(__int128 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rbx
  int v7; // eax
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v10; // r8
  int v11; // r9d
  __int64 v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // rax
  int v15; // r11d
  _QWORD *v16; // rcx
  __int64 v17; // r11
  _QWORD *v18; // rax
  int i; // [rsp+20h] [rbp-80h]
  int v20; // [rsp+24h] [rbp-7Ch]
  __int128 v21; // [rsp+30h] [rbp-70h] BYREF
  __int64 v22; // [rsp+40h] [rbp-60h]
  int v23; // [rsp+50h] [rbp-50h] BYREF
  __m128i v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+68h] [rbp-38h]

  v4 = a1;
  result = sub_159A200((__int64 **)a1, *((__int64 **)&a1 + 1), a2, a3);
  if ( !result )
  {
    v6 = **(_QWORD **)a1;
    v21 = a1;
    v22 = a2;
    v23 = sub_1597240(*((__int64 **)&a1 + 1), *((_QWORD *)&a1 + 1) + 8 * a2);
    v7 = sub_1597910((__int64 *)&v21, &v23);
    v8 = _mm_loadu_si128((const __m128i *)&v21);
    v23 = v7;
    v25 = v22;
    v24 = v8;
    v9 = *(unsigned int *)(v6 + 1576);
    if ( !(_DWORD)v9 )
      return sub_159DC20(v6 + 1552, v4, *((__int64 **)&a1 + 1), a2, (__int64)&v23);
    v10 = *(_QWORD *)(v6 + 1560);
    v11 = v9 - 1;
    v12 = ((_DWORD)v9 - 1) & (unsigned int)v7;
    v13 = (__int64 *)(v10 + 8 * v12);
    v20 = v12;
    v14 = *v13;
    if ( *v13 == -8 )
      return sub_159DC20(v6 + 1552, v4, *((__int64 **)&a1 + 1), a2, (__int64)&v23);
    for ( i = 1; ; ++i )
    {
      if ( v14 != -16 && v24.m128i_i64[0] == *(_QWORD *)v14 )
      {
        v15 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
        if ( v22 == v15 )
        {
          if ( !v15 )
          {
LABEL_15:
            if ( v13 != (__int64 *)(v10 + 8 * v9) )
              return *v13;
            return sub_159DC20(v6 + 1552, v4, *((__int64 **)&a1 + 1), a2, (__int64)&v23);
          }
          v16 = (_QWORD *)v24.m128i_i64[1];
          v17 = v24.m128i_i64[1] + 8 + 8LL * (unsigned int)(v15 - 1);
          v18 = (_QWORD *)(-24 * v22 + v14);
          while ( *v16 == *v18 )
          {
            ++v16;
            v18 += 3;
            if ( (_QWORD *)v17 == v16 )
              goto LABEL_15;
          }
        }
      }
      v13 = (__int64 *)(v10 + 8LL * (v11 & (unsigned int)(v20 + i)));
      v20 = v11 & (v20 + i);
      v14 = *v13;
      if ( *v13 == -8 )
        return sub_159DC20(v6 + 1552, v4, *((__int64 **)&a1 + 1), a2, (__int64)&v23);
    }
  }
  return result;
}
