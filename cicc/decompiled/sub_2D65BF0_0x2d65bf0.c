// Function: sub_2D65BF0
// Address: 0x2d65bf0
//
__int64 __fastcall sub_2D65BF0(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 v4; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  int v8; // edx
  unsigned int v9; // r14d
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  int v12; // edx
  char v13; // cl
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __m128i *v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  unsigned int v23; // r15d
  unsigned __int64 v25; // rax
  unsigned int v26; // r14d
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __m128i *v31; // rax
  __int64 v32; // rdx
  char v33; // [rsp+Fh] [rbp-81h] BYREF
  __m128i v34; // [rsp+10h] [rbp-80h] BYREF
  __m128i v35; // [rsp+20h] [rbp-70h] BYREF
  __m128i v36; // [rsp+30h] [rbp-60h] BYREF
  __m128i v37; // [rsp+40h] [rbp-50h] BYREF
  __int64 v38; // [rsp+50h] [rbp-40h]

  v3 = a3;
  v4 = 0;
  v6 = *(_QWORD *)(a1 + 120);
  v7 = *(unsigned int *)(v6 + 8);
  if ( (_DWORD)v7 )
    v4 = *(_QWORD *)(*(_QWORD *)v6 + 8 * v7 - 8);
  v8 = *a2;
  if ( (_BYTE)v8 != 17 )
  {
    if ( (unsigned __int8)v8 <= 3u )
    {
      v20 = *(__m128i **)(a1 + 96);
      if ( !v20->m128i_i64[0] )
      {
        v20->m128i_i64[0] = (__int64)a2;
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                               + 1288LL))(
               *(_QWORD *)(a1 + 8),
               *(_QWORD *)(a1 + 24),
               *(_QWORD *)(a1 + 96),
               *(_QWORD *)(a1 + 72),
               *(unsigned int *)(a1 + 80),
               0) )
        {
          return 1;
        }
        **(_QWORD **)(a1 + 96) = 0;
        v20 = *(__m128i **)(a1 + 96);
      }
      goto LABEL_22;
    }
    if ( (unsigned __int8)v8 <= 0x1Cu )
    {
      if ( (_BYTE)v8 == 5 )
      {
        if ( (unsigned int)v3 <= 4 )
        {
          if ( (unsigned __int8)sub_2D665B0(a1, a2, *((unsigned __int16 *)a2 + 1), v3, 0) )
            return 1;
          v6 = *(_QWORD *)(a1 + 120);
        }
        sub_2D57BD0((__int64 *)v6, v4);
        v20 = *(__m128i **)(a1 + 96);
        goto LABEL_22;
      }
      if ( (_BYTE)v8 == 20 )
        return 1;
    }
    else
    {
      v20 = *(__m128i **)(a1 + 96);
      v33 = 0;
      v34 = _mm_loadu_si128(v20);
      v35 = _mm_loadu_si128(v20 + 1);
      v36 = _mm_loadu_si128(v20 + 2);
      v37 = _mm_loadu_si128(v20 + 3);
      v38 = v20[4].m128i_i64[0];
      v26 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
      if ( (unsigned int)v3 > 4 )
        goto LABEL_22;
      v23 = sub_2D665B0(a1, a2, (unsigned int)(v8 - 29), v3, &v33);
      if ( (_BYTE)v23 )
      {
        if ( v33 )
          return v23;
        v27 = *((_QWORD *)a2 + 2);
        if ( v27 && !*(_QWORD *)(v27 + 8) || (unsigned __int8)sub_2D674A0(a1, a2, &v34, *(_QWORD *)(a1 + 96)) )
        {
          sub_9C95B0(*(_QWORD *)a1, (__int64)a2);
          return v23;
        }
        v31 = *(__m128i **)(a1 + 96);
        *v31 = _mm_loadu_si128(&v34);
        v31[1] = _mm_loadu_si128(&v35);
        v31[2] = _mm_loadu_si128(&v36);
        v31[3] = _mm_loadu_si128(&v37);
        v32 = (unsigned __int8)v38;
        v31[4].m128i_i8[0] = v38;
        sub_2D65B70(*(_QWORD *)a1, v26, v32, v28, v29, v30);
        sub_2D57BD0(*(__int64 **)(a1 + 120), v4);
        v20 = *(__m128i **)(a1 + 96);
        goto LABEL_22;
      }
    }
    goto LABEL_34;
  }
  v9 = *((_DWORD *)a2 + 8);
  v10 = *((_QWORD *)a2 + 3);
  v11 = 1LL << ((unsigned __int8)v9 - 1);
  if ( v9 > 0x40 )
  {
    v21 = (__int64)(a2 + 24);
    if ( (*(_QWORD *)(v10 + 8LL * ((v9 - 1) >> 6)) & v11) != 0 )
      v22 = sub_C44500(v21);
    else
      v22 = sub_C444A0(v21);
    if ( v9 + 1 - v22 <= 0x40 )
    {
      v16 = *(_QWORD *)v10;
      goto LABEL_11;
    }
    goto LABEL_34;
  }
  if ( (v11 & v10) == 0 )
  {
    if ( !v10 || (_BitScanReverse64(&v25, v10), (unsigned int)v25 ^ 0x3F) )
    {
      v16 = 0;
      if ( !v9 )
        goto LABEL_11;
      v13 = 64 - v9;
      v14 = v10 << (64 - (unsigned __int8)v9);
      goto LABEL_10;
    }
LABEL_34:
    v20 = *(__m128i **)(a1 + 96);
    goto LABEL_22;
  }
  if ( !v9 )
  {
    v16 = 0;
    goto LABEL_11;
  }
  v12 = 64;
  v13 = 64 - v9;
  v14 = v10 << (64 - (unsigned __int8)v9);
  if ( v14 != -1 )
  {
    _BitScanReverse64(&v15, ~(v10 << (64 - (unsigned __int8)v9)));
    v12 = v15 ^ 0x3F;
  }
  if ( v9 + 1 - v12 > 0x40 )
    goto LABEL_34;
LABEL_10:
  v16 = v14 >> v13;
LABEL_11:
  *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL) += v16;
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 1288LL))(
         *(_QWORD *)(a1 + 8),
         *(_QWORD *)(a1 + 24),
         *(_QWORD *)(a1 + 96),
         *(_QWORD *)(a1 + 72),
         *(unsigned int *)(a1 + 80),
         0) )
  {
    return 1;
  }
  v17 = *((_DWORD *)a2 + 8);
  v18 = (__int64 *)*((_QWORD *)a2 + 3);
  if ( v17 > 0x40 )
  {
    v19 = *v18;
  }
  else
  {
    v19 = 0;
    if ( v17 )
      v19 = (__int64)((_QWORD)v18 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL) -= v19;
  v20 = *(__m128i **)(a1 + 96);
LABEL_22:
  if ( !v20[1].m128i_i8[0] )
  {
    v20[1].m128i_i8[0] = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 40LL) = a2;
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                           + 1288LL))(
           *(_QWORD *)(a1 + 8),
           *(_QWORD *)(a1 + 24),
           *(_QWORD *)(a1 + 96),
           *(_QWORD *)(a1 + 72),
           *(unsigned int *)(a1 + 80),
           0) )
    {
      return 1;
    }
    *(_BYTE *)(*(_QWORD *)(a1 + 96) + 16LL) = 0;
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 40LL) = 0;
    v20 = *(__m128i **)(a1 + 96);
  }
  if ( !v20[1].m128i_i64[1] )
  {
    v20[1].m128i_i64[1] = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 48LL) = a2;
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8)
                                                                                            + 1288LL))(
            *(_QWORD *)(a1 + 8),
            *(_QWORD *)(a1 + 24),
            *(_QWORD *)(a1 + 96),
            *(_QWORD *)(a1 + 72),
            *(unsigned int *)(a1 + 80),
            0) )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 96) + 24LL) = 0;
      *(_QWORD *)(*(_QWORD *)(a1 + 96) + 48LL) = 0;
      goto LABEL_28;
    }
    return 1;
  }
LABEL_28:
  v23 = 0;
  sub_2D57BD0(*(__int64 **)(a1 + 120), v4);
  return v23;
}
