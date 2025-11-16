// Function: sub_26446F0
// Address: 0x26446f0
//
__int64 *__fastcall sub_26446F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rcx
  _DWORD *v6; // rax
  __int64 v7; // rsi
  _DWORD *v8; // rdx
  unsigned int *v9; // r14
  unsigned int *v10; // r13
  unsigned __int64 v11; // rax
  unsigned int *v12; // r15
  unsigned int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char *v18[4]; // [rsp+10h] [rbp-140h] BYREF
  __m128i v19[2]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v20; // [rsp+50h] [rbp-100h]
  __m128i v21[2]; // [rsp+60h] [rbp-F0h] BYREF
  char v22; // [rsp+80h] [rbp-D0h]
  char v23; // [rsp+81h] [rbp-CFh]
  __m128i v24; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+A0h] [rbp-B0h]
  __m128i v26; // [rsp+C0h] [rbp-90h] BYREF
  _QWORD v27[2]; // [rsp+D0h] [rbp-80h] BYREF
  char v28; // [rsp+E0h] [rbp-70h]
  char v29; // [rsp+E1h] [rbp-6Fh]
  __m128i v30; // [rsp+F0h] [rbp-60h] BYREF
  _DWORD *v31; // [rsp+100h] [rbp-50h]
  _DWORD *v32; // [rsp+108h] [rbp-48h]
  __int16 v33; // [rsp+110h] [rbp-40h]

  sub_263F570(a1, "ContextIds:");
  v5 = *(unsigned int *)(a2 + 16);
  if ( (unsigned int)v5 > 0x63 )
  {
    v29 = 1;
    v26.m128i_i64[0] = (__int64)" ids)";
    v20 = 265;
    v19[0].m128i_i32[0] = v5;
    v21[0].m128i_i64[0] = (__int64)" (";
    v28 = 3;
    v23 = 1;
    v22 = 3;
    sub_9C6370(&v24, v21, v19, v5, v3, v4);
    sub_9C6370(&v30, &v24, &v26, v14, v15, v16);
    sub_CA0F50((__int64 *)v18, (void **)&v30);
    sub_2241490((unsigned __int64 *)a1, v18[0], (size_t)v18[1]);
    sub_2240A30((unsigned __int64 *)v18);
  }
  else
  {
    v6 = *(_DWORD **)(a2 + 8);
    v7 = *(_QWORD *)a2;
    v8 = &v6[*(unsigned int *)(a2 + 24)];
    if ( (_DWORD)v5 )
    {
      for ( ; v8 != v6; ++v6 )
      {
        if ( *v6 <= 0xFFFFFFFD )
          break;
      }
    }
    else
    {
      v6 += *(unsigned int *)(a2 + 24);
    }
    v26.m128i_i64[0] = a2;
    v26.m128i_i64[1] = v7;
    v27[0] = v6;
    v27[1] = v8;
    v30.m128i_i64[0] = a2;
    v30.m128i_i64[1] = v7;
    v31 = v8;
    v32 = v8;
    v24 = 0u;
    v25 = 0;
    sub_2641230(v24.m128i_i64, v7, (__int64)v8, v5, v3, v4, a2, v7, v6, v8, a2, v7, v8);
    v9 = (unsigned int *)v24.m128i_i64[1];
    v10 = (unsigned int *)v24.m128i_i64[0];
    if ( v24.m128i_i64[1] != v24.m128i_i64[0] )
    {
      _BitScanReverse64(&v11, (v24.m128i_i64[1] - v24.m128i_i64[0]) >> 2);
      sub_263F8F0((char *)v24.m128i_i64[0], (char *)v24.m128i_i64[1], 2LL * (int)(63 - (v11 ^ 0x3F)));
      sub_263F470(v10, v9);
      v10 = (unsigned int *)v24.m128i_i64[1];
      if ( v24.m128i_i64[1] != v24.m128i_i64[0] )
      {
        v12 = (unsigned int *)v24.m128i_i64[0];
        do
        {
          v13 = *v12;
          v30.m128i_i64[0] = (__int64)" ";
          LODWORD(v31) = v13;
          v33 = 2307;
          sub_CA0F50(v26.m128i_i64, (void **)&v30);
          sub_2241490((unsigned __int64 *)a1, (char *)v26.m128i_i64[0], v26.m128i_u64[1]);
          if ( (_QWORD *)v26.m128i_i64[0] != v27 )
            j_j___libc_free_0(v26.m128i_u64[0]);
          ++v12;
        }
        while ( v10 != v12 );
        v10 = (unsigned int *)v24.m128i_i64[0];
      }
    }
    if ( v10 )
      j_j___libc_free_0((unsigned __int64)v10);
  }
  return a1;
}
