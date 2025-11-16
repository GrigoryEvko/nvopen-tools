// Function: sub_375D5B0
// Address: 0x375d5b0
//
__int64 __fastcall sub_375D5B0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v5; // edi
  char v7; // dl
  __int64 v8; // rsi
  int v9; // ecx
  int v10; // r9d
  unsigned int i; // eax
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 result; // rax
  int v17; // eax
  char v18; // di
  unsigned int v19; // r14d
  int v20; // edi
  unsigned int v21; // esi
  __int64 v22; // r8
  int v23; // esi
  unsigned int v24; // edx
  unsigned int *v25; // rcx
  __int64 v26; // r14
  unsigned int v27; // eax
  unsigned int *v28; // r9
  int v29; // edx
  unsigned int v30; // edi
  __m128i v31; // xmm0
  int v32; // r10d
  __int64 v33; // rsi
  int v34; // eax
  unsigned int v35; // edx
  unsigned int v36; // ecx
  __int64 v37; // rsi
  int v38; // edx
  unsigned int v39; // ecx
  unsigned int v40; // eax
  int v41; // r8d
  unsigned int *v42; // rdi
  int v43; // eax
  int v44; // edx
  int v45; // r8d
  unsigned __int64 v46[2]; // [rsp+0h] [rbp-70h] BYREF
  int v47[4]; // [rsp+10h] [rbp-60h] BYREF
  char v48[8]; // [rsp+20h] [rbp-50h] BYREF
  __m128i v49; // [rsp+28h] [rbp-48h] BYREF

  v5 = a3;
  v7 = *(_BYTE *)(a1 + 304) & 1;
  if ( v7 )
  {
    v8 = a1 + 312;
    v9 = 7;
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 320);
    v8 = *(_QWORD *)(a1 + 312);
    if ( !(_DWORD)v14 )
    {
LABEL_21:
      v26 = 24 * v14;
LABEL_22:
      v12 = v8 + v26;
      goto LABEL_10;
    }
    v9 = v14 - 1;
  }
  v10 = 1;
  for ( i = v9 & (a3 + ((a2 >> 9) ^ (a2 >> 4))); ; i = v9 & v13 )
  {
    v12 = v8 + 24LL * i;
    if ( *(_QWORD *)v12 == a2 && *(_DWORD *)(v12 + 8) == v5 )
      break;
    if ( !*(_QWORD *)v12 && *(_DWORD *)(v12 + 8) == -1 )
    {
      if ( !v7 )
      {
        v14 = *(unsigned int *)(a1 + 320);
        goto LABEL_21;
      }
      v26 = 192;
      goto LABEL_22;
    }
    v13 = v10 + i;
    ++v10;
  }
LABEL_10:
  v15 = 192;
  if ( !v7 )
    v15 = 24LL * *(unsigned int *)(a1 + 320);
  if ( v12 != v8 + v15 )
  {
    sub_37593F0(a1, (int *)(v12 + 16));
    return *(unsigned int *)(v12 + 16);
  }
  v17 = *(_DWORD *)(a1 + 292);
  v46[0] = a2;
  v46[1] = a3;
  v47[0] = v17;
  sub_375CDA0((__int64)v48, (__m128i *)(a1 + 296), v46, v47);
  v49.m128i_i64[0] = a2;
  v18 = *(_BYTE *)(a1 + 512);
  v49.m128i_i64[1] = a3;
  v19 = *(_DWORD *)(a1 + 292);
  v20 = v18 & 1;
  if ( v20 )
  {
    v22 = a1 + 520;
    v23 = 7;
  }
  else
  {
    v21 = *(_DWORD *)(a1 + 528);
    v22 = *(_QWORD *)(a1 + 520);
    if ( !v21 )
    {
      v27 = *(_DWORD *)(a1 + 512);
      v28 = 0;
      ++*(_QWORD *)(a1 + 504);
      v29 = (v27 >> 1) + 1;
      goto LABEL_24;
    }
    v23 = v21 - 1;
  }
  v24 = v23 & (37 * v19);
  v25 = (unsigned int *)(v22 + 24LL * v24);
  result = *v25;
  if ( v19 != (_DWORD)result )
  {
    v32 = 1;
    v28 = 0;
    while ( (_DWORD)result != -1 )
    {
      if ( (_DWORD)result != -2 || v28 )
        v25 = v28;
      v24 = v23 & (v32 + v24);
      result = *(unsigned int *)(v22 + 24LL * v24);
      if ( v19 == (_DWORD)result )
        goto LABEL_18;
      ++v32;
      v28 = v25;
      v25 = (unsigned int *)(v22 + 24LL * v24);
    }
    v27 = *(_DWORD *)(a1 + 512);
    if ( !v28 )
      v28 = v25;
    ++*(_QWORD *)(a1 + 504);
    v29 = (v27 >> 1) + 1;
    if ( (_BYTE)v20 )
    {
      v30 = 24;
      v21 = 8;
LABEL_25:
      if ( 4 * v29 < v30 )
      {
        if ( v21 - *(_DWORD *)(a1 + 516) - v29 > v21 >> 3 )
        {
LABEL_27:
          *(_DWORD *)(a1 + 512) = (2 * (v27 >> 1) + 2) | v27 & 1;
          if ( *v28 != -1 )
            --*(_DWORD *)(a1 + 516);
          v49.m128i_i64[0] = a2;
          v49.m128i_i32[2] = a3;
          v31 = _mm_loadu_si128(&v49);
          *v28 = v19;
          *(__m128i *)(v28 + 2) = v31;
          result = *(unsigned int *)(a1 + 292);
          goto LABEL_18;
        }
        sub_375D160(a1 + 504, v21);
        if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
        {
          v37 = a1 + 520;
          v38 = 7;
          goto LABEL_45;
        }
        v44 = *(_DWORD *)(a1 + 528);
        v37 = *(_QWORD *)(a1 + 520);
        if ( v44 )
        {
          v38 = v44 - 1;
LABEL_45:
          v39 = v38 & (37 * v19);
          v28 = (unsigned int *)(v37 + 24LL * v39);
          v40 = *v28;
          if ( v19 != *v28 )
          {
            v41 = 1;
            v42 = 0;
            while ( v40 != -1 )
            {
              if ( v40 == -2 && !v42 )
                v42 = v28;
              v39 = v38 & (v41 + v39);
              v28 = (unsigned int *)(v37 + 24LL * v39);
              v40 = *v28;
              if ( v19 == *v28 )
                goto LABEL_42;
              ++v41;
            }
LABEL_48:
            if ( v42 )
              v28 = v42;
            goto LABEL_42;
          }
          goto LABEL_42;
        }
LABEL_72:
        *(_DWORD *)(a1 + 512) = (2 * (*(_DWORD *)(a1 + 512) >> 1) + 2) | *(_DWORD *)(a1 + 512) & 1;
        BUG();
      }
      sub_375D160(a1 + 504, 2 * v21);
      if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
      {
        v33 = a1 + 520;
        v34 = 7;
      }
      else
      {
        v43 = *(_DWORD *)(a1 + 528);
        v33 = *(_QWORD *)(a1 + 520);
        if ( !v43 )
          goto LABEL_72;
        v34 = v43 - 1;
      }
      v35 = v34 & (37 * v19);
      v28 = (unsigned int *)(v33 + 24LL * v35);
      v36 = *v28;
      if ( v19 != *v28 )
      {
        v45 = 1;
        v42 = 0;
        while ( v36 != -1 )
        {
          if ( v36 == -2 && !v42 )
            v42 = v28;
          v35 = v34 & (v45 + v35);
          v28 = (unsigned int *)(v33 + 24LL * v35);
          v36 = *v28;
          if ( v19 == *v28 )
            goto LABEL_42;
          ++v45;
        }
        goto LABEL_48;
      }
LABEL_42:
      v27 = *(_DWORD *)(a1 + 512);
      goto LABEL_27;
    }
    v21 = *(_DWORD *)(a1 + 528);
LABEL_24:
    v30 = 3 * v21;
    goto LABEL_25;
  }
LABEL_18:
  *(_DWORD *)(a1 + 292) = result + 1;
  return result;
}
