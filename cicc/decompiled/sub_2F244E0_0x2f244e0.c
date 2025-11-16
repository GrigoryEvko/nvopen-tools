// Function: sub_2F244E0
// Address: 0x2f244e0
//
void __fastcall sub_2F244E0(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // r15
  _QWORD *v5; // r14
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // edi
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rdx
  char *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __m128i *v20; // rdi
  __m128i v21; // xmm0
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned __int64 *v24; // [rsp+0h] [rbp-B0h]
  __int64 v25[2]; // [rsp+20h] [rbp-90h] BYREF
  __m128i v26; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v27; // [rsp+40h] [rbp-70h] BYREF
  __m128i *v28; // [rsp+48h] [rbp-68h]
  __int64 v29; // [rsp+50h] [rbp-60h]
  __m128i v30; // [rsp+58h] [rbp-58h] BYREF
  __m128i v31; // [rsp+68h] [rbp-48h] BYREF
  int v32; // [rsp+78h] [rbp-38h]

  v4 = *(_QWORD **)(a3 + 728);
  v5 = &v4[3 * *(unsigned int *)(a3 + 744)];
  if ( *(_DWORD *)(a3 + 736) && v4 != v5 )
  {
    while ( 1 )
    {
      v11 = *v4;
      if ( *v4 != -8192 && v11 != -4096 )
        break;
      v4 += 3;
      if ( v5 == v4 )
        goto LABEL_2;
    }
    if ( v5 != v4 )
    {
      v24 = a2 + 73;
      v12 = *(_QWORD *)(v11 + 24);
      v13 = *(_DWORD *)(v12 + 24);
      v14 = *(_QWORD *)(v12 + 56);
      if ( v11 == v14 )
        goto LABEL_35;
LABEL_15:
      v15 = 0;
      do
      {
        v14 = *(_QWORD *)(v14 + 8);
        ++v15;
      }
      while ( v11 != v14 );
      while ( 1 )
      {
        v27 = __PAIR64__(v15, v13);
        v17 = (char *)sub_BD5D20(v4[1]);
        if ( !v17 )
          break;
        v25[0] = (__int64)&v26;
        sub_2F07580(v25, v17, (__int64)&v17[v16]);
        v28 = &v30;
        v18 = v25[1];
        if ( (__m128i *)v25[0] == &v26 )
          goto LABEL_37;
        v28 = (__m128i *)v25[0];
        v30.m128i_i64[0] = v26.m128i_i64[0];
LABEL_20:
        v29 = v18;
        v19 = a2[74];
        v31 = 0u;
        v32 = *((_DWORD *)v4 + 4);
        if ( v19 == a2[75] )
        {
          sub_2F18F40(v24, (const __m128i *)v19, (__int64)&v27);
          v20 = v28;
        }
        else
        {
          if ( v19 )
          {
            *(_QWORD *)v19 = v27;
            *(_QWORD *)(v19 + 8) = v19 + 24;
            if ( v28 == &v30 )
            {
              *(__m128i *)(v19 + 24) = _mm_loadu_si128(&v30);
            }
            else
            {
              *(_QWORD *)(v19 + 8) = v28;
              *(_QWORD *)(v19 + 24) = v30.m128i_i64[0];
            }
            v20 = &v30;
            v28 = &v30;
            *(_QWORD *)(v19 + 16) = v29;
            v21 = _mm_loadu_si128(&v31);
            v29 = 0;
            *(__m128i *)(v19 + 40) = v21;
            v30.m128i_i8[0] = 0;
            *(_DWORD *)(v19 + 56) = v32;
            v19 = a2[74];
          }
          else
          {
            v20 = v28;
          }
          a2[74] = v19 + 64;
        }
        if ( v20 != &v30 )
          j_j___libc_free_0((unsigned __int64)v20);
        v22 = v4 + 3;
        if ( v4 + 3 == v5 )
          goto LABEL_2;
        while ( 1 )
        {
          v11 = *v22;
          v4 = v22;
          if ( *v22 != -4096 && v11 != -8192 )
            break;
          v22 += 3;
          if ( v5 == v22 )
            goto LABEL_2;
        }
        if ( v5 == v22 )
          goto LABEL_2;
        v23 = *(_QWORD *)(v11 + 24);
        v13 = *(_DWORD *)(v23 + 24);
        v14 = *(_QWORD *)(v23 + 56);
        if ( v11 != v14 )
          goto LABEL_15;
LABEL_35:
        v15 = 0;
      }
      v26.m128i_i8[0] = 0;
      v18 = 0;
      v28 = &v30;
LABEL_37:
      v30 = _mm_load_si128(&v26);
      goto LABEL_20;
    }
  }
LABEL_2:
  v6 = a2[74];
  v7 = a2[73];
  if ( v7 != v6 )
  {
    _BitScanReverse64(&v8, (v6 - v7) >> 6);
    sub_2F239C0(a2[73], (__int8 *)a2[74], 2LL * (int)(63 - (v8 ^ 0x3F)));
    if ( v6 - v7 <= 1024 )
    {
      sub_2F0D7B0(v7, v6);
    }
    else
    {
      v9 = (__int64 *)(v7 + 1024);
      sub_2F0D7B0(v7, v7 + 1024);
      if ( v6 != v7 + 1024 )
      {
        do
        {
          v10 = v9;
          v9 += 8;
          sub_2F0D3A0(v10);
        }
        while ( (__int64 *)v6 != v9 );
      }
    }
  }
}
