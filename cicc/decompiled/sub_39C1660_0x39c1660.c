// Function: sub_39C1660
// Address: 0x39c1660
//
__int64 __fastcall sub_39C1660(__int64 a1, const __m128i *a2)
{
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // esi
  __int64 v7; // rdx
  int v8; // r9d
  int v9; // r10d
  unsigned int v10; // edi
  __int64 *v11; // r15
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  unsigned int i; // r8d
  __int64 *v16; // rcx
  __int64 v17; // rdi
  unsigned int v18; // r8d
  __int64 v19; // rax
  int v21; // edx
  __int64 v22; // rcx
  __m128i v23; // xmm0
  __m128i *v24; // rsi
  _BYTE *v25; // rdi
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // rdx
  int v29; // r8d
  __int64 *v30; // rdi
  unsigned __int64 v31; // rsi
  unsigned __int64 v32; // rsi
  unsigned int j; // eax
  __int64 v34; // rsi
  unsigned int v35; // eax
  int v36; // edx
  int v37; // edx
  __int64 v38; // rsi
  int v39; // r8d
  unsigned int k; // eax
  __int64 v41; // rcx
  unsigned int v42; // eax
  int v43; // [rsp+8h] [rbp-E8h]
  __m128i v44; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v45; // [rsp+70h] [rbp-80h] BYREF
  __int64 v46; // [rsp+78h] [rbp-78h]
  _BYTE v47[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = a2->m128i_i64[0];
  v5 = a2->m128i_i64[1];
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_33;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v6 - 1;
  v9 = 1;
  v10 = (unsigned int)v5 >> 9;
  v11 = 0;
  v12 = (((v10 ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v10 ^ ((unsigned int)v5 >> 4)) << 32)) >> 22)
      ^ ((v10 ^ ((unsigned int)v5 >> 4) | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v10 ^ ((unsigned int)v5 >> 4)) << 32));
  v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
      ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
  v14 = ((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - (v13 << 27));
  for ( i = v14 & (v6 - 1); ; i = v8 & v18 )
  {
    v16 = (__int64 *)(v7 + 24LL * i);
    v17 = *v16;
    if ( v4 == *v16 && v16[1] == v5 )
    {
      v19 = *((unsigned int *)v16 + 4);
      return *(_QWORD *)(a1 + 32) + 96 * v19 + 16;
    }
    if ( v17 == -8 )
      break;
    if ( v17 == -16 && v16[1] == -16 && !v11 )
      v11 = (__int64 *)(v7 + 24LL * i);
LABEL_9:
    v18 = v9 + i;
    ++v9;
  }
  if ( v16[1] != -8 )
    goto LABEL_9;
  v21 = *(_DWORD *)(a1 + 16);
  if ( !v11 )
    v11 = v16;
  ++*(_QWORD *)a1;
  v22 = (unsigned int)(v21 + 1);
  if ( 4 * (int)v22 >= 3 * v6 )
  {
LABEL_33:
    sub_39C13B0(a1, 2 * v6);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v29 = 1;
      v30 = 0;
      v31 = (((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
             | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
            | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32));
      v32 = ((9 * (((v31 - 1 - (v31 << 13)) >> 8) ^ (v31 - 1 - (v31 << 13)))) >> 15)
          ^ (9 * (((v31 - 1 - (v31 << 13)) >> 8) ^ (v31 - 1 - (v31 << 13))));
      for ( j = v27 & (((v32 - 1 - (v32 << 27)) >> 31) ^ (v32 - 1 - ((_DWORD)v32 << 27))); ; j = v27 & v35 )
      {
        v28 = *(_QWORD *)(a1 + 8);
        v11 = (__int64 *)(v28 + 24LL * j);
        v34 = *v11;
        if ( v4 == *v11 && v11[1] == v5 )
          break;
        if ( v34 == -8 )
        {
          if ( v11[1] == -8 )
          {
LABEL_56:
            if ( v30 )
              v11 = v30;
            v22 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
            goto LABEL_18;
          }
        }
        else if ( v34 == -16 && v11[1] == -16 && !v30 )
        {
          v30 = (__int64 *)(v28 + 24LL * j);
        }
        v35 = v29 + j;
        ++v29;
      }
      goto LABEL_52;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - (unsigned int)v22 <= v6 >> 3 )
  {
    v43 = v14;
    sub_39C13B0(a1, v6);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v30 = 0;
      v39 = 1;
      for ( k = v37 & v43; ; k = v37 & v42 )
      {
        v38 = *(_QWORD *)(a1 + 8);
        v11 = (__int64 *)(v38 + 24LL * k);
        v41 = *v11;
        if ( v4 == *v11 && v11[1] == v5 )
          break;
        if ( v41 == -8 )
        {
          if ( v11[1] == -8 )
            goto LABEL_56;
        }
        else if ( v41 == -16 && v11[1] == -16 && !v30 )
        {
          v30 = (__int64 *)(v38 + 24LL * k);
        }
        v42 = v39 + k;
        ++v39;
      }
LABEL_52:
      v22 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
      goto LABEL_18;
    }
    goto LABEL_61;
  }
LABEL_18:
  *(_DWORD *)(a1 + 16) = v22;
  if ( *v11 != -8 || v11[1] != -8 )
    --*(_DWORD *)(a1 + 20);
  *v11 = v4;
  v11[1] = v5;
  *((_DWORD *)v11 + 4) = 0;
  v23 = _mm_loadu_si128(a2);
  v24 = *(__m128i **)(a1 + 40);
  v45 = v47;
  v46 = 0x400000000LL;
  v44 = v23;
  if ( v24 == *(__m128i **)(a1 + 48) )
  {
    sub_39C0F80((unsigned __int64 *)(a1 + 32), v24, &v44, v22);
    v25 = v45;
  }
  else
  {
    v25 = v47;
    if ( v24 )
    {
      v24[1].m128i_i64[1] = 0x400000000LL;
      v24[1].m128i_i64[0] = (__int64)v24[2].m128i_i64;
      *v24 = v23;
      if ( (_DWORD)v46 )
        sub_39C0A30((__int64)v24[1].m128i_i64, (__int64)&v45, (__int64)v24[2].m128i_i64, v22, (int)&v45, v8);
      v24 = *(__m128i **)(a1 + 40);
      v25 = v45;
    }
    *(_QWORD *)(a1 + 40) = v24 + 6;
  }
  if ( v25 != v47 )
    _libc_free((unsigned __int64)v25);
  v19 = -1431655765 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 5) - 1;
  *((_DWORD *)v11 + 4) = v19;
  return *(_QWORD *)(a1 + 32) + 96 * v19 + 16;
}
