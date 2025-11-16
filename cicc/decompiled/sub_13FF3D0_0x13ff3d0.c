// Function: sub_13FF3D0
// Address: 0x13ff3d0
//
__int64 __fastcall sub_13FF3D0(__int128 a1)
{
  __int64 v1; // rax
  const __m128i *v2; // rbx
  __int64 v3; // rdi
  _BYTE *v4; // rsi
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 result; // rax
  int v15; // r15d
  __int64 *v16; // r11
  int v17; // edx
  int v18; // edi
  int v19; // eax
  __int64 v20; // r8
  int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // edx
  int v24; // r11d
  __int64 *v25; // r10
  int v26; // eax
  int v27; // esi
  __int64 v28; // r9
  int v29; // r11d
  unsigned int v30; // edx
  __int64 v31; // [rsp+18h] [rbp-68h] BYREF
  __int128 v32; // [rsp+20h] [rbp-60h] BYREF
  const __m128i *v33; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h]
  __int64 v36; // [rsp+48h] [rbp-38h]

  v32 = a1;
  sub_13FF050((__int64 *)&v33, (const __m128i *)&v32);
LABEL_10:
  v13 = v34;
  result = v35;
  while ( v13 != result )
  {
    v1 = *(_QWORD *)(result - 24);
    v2 = v33;
    v31 = v1;
    v3 = v33->m128i_i64[0];
    v4 = *(_BYTE **)(v33->m128i_i64[0] + 48);
    if ( v4 == *(_BYTE **)(v33->m128i_i64[0] + 56) )
    {
      sub_1292090(v3 + 40, v4, &v31);
    }
    else
    {
      if ( v4 )
      {
        *(_QWORD *)v4 = v1;
        v4 = *(_BYTE **)(v3 + 48);
      }
      *(_QWORD *)(v3 + 48) = v4 + 8;
    }
    v5 = v2->m128i_i64[0];
    v6 = *(_DWORD *)(v5 + 32);
    v7 = (__int64)(*(_QWORD *)(v5 + 48) - *(_QWORD *)(v5 + 40)) >> 3;
    if ( !v6 )
    {
      ++*(_QWORD *)(v5 + 8);
      goto LABEL_26;
    }
    v8 = v31;
    v9 = *(_QWORD *)(v5 + 16);
    v10 = (v6 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v31 != *v11 )
    {
      v15 = 1;
      v16 = 0;
      while ( v12 != -8 )
      {
        if ( v12 == -16 && !v16 )
          v16 = v11;
        v10 = (v6 - 1) & (v15 + v10);
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v31 == *v11 )
          goto LABEL_8;
        ++v15;
      }
      v17 = *(_DWORD *)(v5 + 24);
      if ( v16 )
        v11 = v16;
      ++*(_QWORD *)(v5 + 8);
      v18 = v17 + 1;
      if ( 4 * (v17 + 1) >= 3 * v6 )
      {
LABEL_26:
        sub_13FEAC0(v5 + 8, 2 * v6);
        v19 = *(_DWORD *)(v5 + 32);
        if ( !v19 )
          goto LABEL_51;
        v20 = v31;
        v21 = v19 - 1;
        v22 = *(_QWORD *)(v5 + 16);
        v18 = *(_DWORD *)(v5 + 24) + 1;
        v23 = (v19 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v11 = (__int64 *)(v22 + 16LL * v23);
        v8 = *v11;
        if ( v31 != *v11 )
        {
          v24 = 1;
          v25 = 0;
          while ( v8 != -8 )
          {
            if ( !v25 && v8 == -16 )
              v25 = v11;
            v23 = v21 & (v24 + v23);
            v11 = (__int64 *)(v22 + 16LL * v23);
            v8 = *v11;
            if ( v31 == *v11 )
              goto LABEL_22;
            ++v24;
          }
LABEL_30:
          v8 = v20;
          if ( v25 )
            v11 = v25;
        }
      }
      else if ( v6 - *(_DWORD *)(v5 + 28) - v18 <= v6 >> 3 )
      {
        sub_13FEAC0(v5 + 8, v6);
        v26 = *(_DWORD *)(v5 + 32);
        if ( !v26 )
        {
LABEL_51:
          ++*(_DWORD *)(v5 + 24);
          BUG();
        }
        v20 = v31;
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v5 + 16);
        v25 = 0;
        v29 = 1;
        v18 = *(_DWORD *)(v5 + 24) + 1;
        v30 = (v26 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v11 = (__int64 *)(v28 + 16LL * v30);
        v8 = *v11;
        if ( v31 != *v11 )
        {
          while ( v8 != -8 )
          {
            if ( v8 == -16 && !v25 )
              v25 = v11;
            v30 = v27 & (v29 + v30);
            v11 = (__int64 *)(v28 + 16LL * v30);
            v8 = *v11;
            if ( v31 == *v11 )
              goto LABEL_22;
            ++v29;
          }
          goto LABEL_30;
        }
      }
LABEL_22:
      *(_DWORD *)(v5 + 24) = v18;
      if ( *v11 != -8 )
        --*(_DWORD *)(v5 + 28);
      *v11 = v8;
      *((_DWORD *)v11 + 2) = 0;
    }
LABEL_8:
    *((_DWORD *)v11 + 2) = v7;
    v13 = v34;
    v35 -= 24;
    result = v35;
    if ( v35 != v34 )
    {
      sub_13FEC80(&v33);
      goto LABEL_10;
    }
  }
  if ( v13 )
    return j_j___libc_free_0(v13, v36 - v13);
  return result;
}
