// Function: sub_20F1860
// Address: 0x20f1860
//
__int64 __fastcall sub_20F1860(__int64 a1, const __m128i *a2)
{
  __int32 v4; // r13d
  __int64 v5; // r14
  unsigned int v6; // esi
  __int64 v7; // rdx
  int v8; // r10d
  __int32 *v9; // r15
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int i; // r8d
  int *v14; // rcx
  int v15; // edi
  unsigned int v16; // r8d
  __int64 v17; // rax
  int v19; // edx
  int v20; // ecx
  __m128i v21; // xmm0
  const __m128i *v22; // rsi
  __m128i *v23; // rdi
  __int64 v24; // rsi
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // rdx
  int v28; // r8d
  int *v29; // rdi
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rsi
  unsigned int j; // eax
  int v33; // esi
  unsigned int v34; // eax
  int v35; // edx
  int v36; // edx
  __int64 v37; // rsi
  int v38; // r8d
  unsigned int k; // eax
  int v40; // ecx
  unsigned int v41; // eax
  int v42; // [rsp+8h] [rbp-1A8h]
  __int64 v43; // [rsp+10h] [rbp-1A0h] BYREF
  _BYTE *v44; // [rsp+18h] [rbp-198h]
  _BYTE *v45; // [rsp+20h] [rbp-190h]
  __int64 v46; // [rsp+28h] [rbp-188h]
  int v47; // [rsp+30h] [rbp-180h]
  _BYTE v48[136]; // [rsp+38h] [rbp-178h] BYREF
  __m128i v49; // [rsp+C0h] [rbp-F0h] BYREF
  _QWORD v50[2]; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned __int64 v51; // [rsp+E0h] [rbp-D0h]
  _BYTE v52[184]; // [rsp+F8h] [rbp-B8h] BYREF

  v4 = a2->m128i_i32[0];
  v5 = a2->m128i_i64[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_33;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  v12 = ((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - (v11 << 27));
  for ( i = v12 & (v6 - 1); ; i = (v6 - 1) & v16 )
  {
    v14 = (int *)(v7 + 24LL * i);
    v15 = *v14;
    if ( v4 == *v14 && *((_QWORD *)v14 + 1) == v5 )
    {
      v17 = (unsigned int)v14[4];
      return *(_QWORD *)(a1 + 32) + 184 * v17 + 16;
    }
    if ( v15 == 0x7FFFFFFF )
      break;
    if ( v15 == 0x80000000 && *((_QWORD *)v14 + 1) == -16 && !v9 )
      v9 = (__int32 *)(v7 + 24LL * i);
LABEL_9:
    v16 = v8 + i;
    ++v8;
  }
  if ( *((_QWORD *)v14 + 1) != -8 )
    goto LABEL_9;
  v19 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v14;
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_33:
    sub_20F15B0(a1, 2 * v6);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v28 = 1;
      v29 = 0;
      v30 = (((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32));
      v31 = ((9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)))) >> 15)
          ^ (9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13))));
      for ( j = v26 & (((v31 - 1 - (v31 << 27)) >> 31) ^ (v31 - 1 - ((_DWORD)v31 << 27))); ; j = v26 & v34 )
      {
        v27 = *(_QWORD *)(a1 + 8);
        v9 = (__int32 *)(v27 + 24LL * j);
        v33 = *v9;
        if ( v4 == *v9 && *((_QWORD *)v9 + 1) == v5 )
          break;
        if ( v33 == 0x7FFFFFFF )
        {
          if ( *((_QWORD *)v9 + 1) == -8 )
          {
LABEL_56:
            if ( v29 )
              v9 = v29;
            v20 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_18;
          }
        }
        else if ( v33 == 0x80000000 && *((_QWORD *)v9 + 1) == -16 && !v29 )
        {
          v29 = (int *)(v27 + 24LL * j);
        }
        v34 = v28 + j;
        ++v28;
      }
      goto LABEL_52;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v20 <= v6 >> 3 )
  {
    v42 = v12;
    sub_20F15B0(a1, v6);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v29 = 0;
      v38 = 1;
      for ( k = v36 & v42; ; k = v36 & v41 )
      {
        v37 = *(_QWORD *)(a1 + 8);
        v9 = (__int32 *)(v37 + 24LL * k);
        v40 = *v9;
        if ( v4 == *v9 && v5 == *((_QWORD *)v9 + 1) )
          break;
        if ( v40 == 0x7FFFFFFF )
        {
          if ( *((_QWORD *)v9 + 1) == -8 )
            goto LABEL_56;
        }
        else if ( v40 == 0x80000000 && *((_QWORD *)v9 + 1) == -16 && !v29 )
        {
          v29 = (int *)(v37 + 24LL * k);
        }
        v41 = v38 + k;
        ++v38;
      }
LABEL_52:
      v20 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_18;
    }
    goto LABEL_61;
  }
LABEL_18:
  *(_DWORD *)(a1 + 16) = v20;
  if ( *v9 != 0x7FFFFFFF || *((_QWORD *)v9 + 1) != -8 )
    --*(_DWORD *)(a1 + 20);
  *v9 = v4;
  *((_QWORD *)v9 + 1) = v5;
  v9[4] = 0;
  v21 = _mm_loadu_si128(a2);
  v44 = v48;
  v43 = 0;
  v45 = v48;
  v46 = 16;
  v47 = 0;
  v49 = v21;
  sub_16CCEE0(v50, (__int64)v52, 16, (__int64)&v43);
  v22 = *(const __m128i **)(a1 + 40);
  if ( v22 == *(const __m128i **)(a1 + 48) )
  {
    sub_20EB6D0((const __m128i **)(a1 + 32), v22, &v49);
  }
  else
  {
    if ( v22 )
    {
      v23 = (__m128i *)&v22[1];
      v24 = (__int64)&v22[3].m128i_i64[1];
      *(__m128i *)(v24 - 56) = _mm_loadu_si128(&v49);
      sub_16CCEE0(v23, v24, 16, (__int64)v50);
      v22 = *(const __m128i **)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = (char *)v22 + 184;
  }
  if ( v51 != v50[1] )
    _libc_free(v51);
  if ( v45 != v44 )
    _libc_free((unsigned __int64)v45);
  v17 = -373475417 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  v9[4] = v17;
  return *(_QWORD *)(a1 + 32) + 184 * v17 + 16;
}
