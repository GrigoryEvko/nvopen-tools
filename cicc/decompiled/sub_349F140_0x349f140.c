// Function: sub_349F140
// Address: 0x349f140
//
__int64 __fastcall sub_349F140(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  bool v9; // zf
  char *v10; // r9
  __int64 result; // rax
  char *i; // r14
  int v13; // eax
  __int64 v14; // r10
  unsigned __int64 v15; // r13
  __m128i v16; // xmm0
  __m128i *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // r10
  __m128i *v21; // r11
  char v22; // al
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  const __m128i *v25; // r11
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdi
  const void *v30; // [rsp+8h] [rbp-98h]
  const void *v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+28h] [rbp-78h]
  __m128i v34; // [rsp+30h] [rbp-70h] BYREF
  __int64 v35; // [rsp+40h] [rbp-60h]
  int v36; // [rsp+50h] [rbp-50h] BYREF
  __m128i v37; // [rsp+58h] [rbp-48h]
  __int64 v38; // [rsp+68h] [rbp-38h]

  v4 = sub_B10CD0(a2 + 56);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
  {
    if ( *(_DWORD *)(v4 - 24) != 2 )
    {
LABEL_3:
      v6 = 0;
      goto LABEL_4;
    }
    v23 = *(_QWORD *)(v4 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v23 = v4 - 16 - 8LL * ((v5 >> 2) & 0xF);
  }
  v6 = *(_QWORD *)(v23 + 8);
LABEL_4:
  v7 = sub_2E891C0(a2);
  *(_QWORD *)a1 = sub_2E89170(a2);
  if ( v7 )
    sub_AF47B0(a1 + 8, *(unsigned __int64 **)(v7 + 16), *(unsigned __int64 **)(v7 + 24));
  else
    *(_BYTE *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = v6;
  v8 = sub_2E891C0(a2);
  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  v31 = (const void *)(a1 + 80);
  *(_QWORD *)(a1 + 72) = 0x800000000LL;
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 344) = 0x800000000LL;
  v9 = *(_WORD *)(a2 + 68) == 14;
  *(_DWORD *)(a1 + 56) = 0;
  v30 = (const void *)(a1 + 352);
  v10 = *(char **)(a2 + 32);
  if ( v9 )
  {
    result = (__int64)(v10 + 40);
    v33 = (__int64)(v10 + 40);
  }
  else
  {
    result = (__int64)&v10[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
    v10 += 80;
    v33 = result;
  }
  for ( i = v10; (char *)v33 != i; ++*(_DWORD *)(a1 + 344) )
  {
    while ( 1 )
    {
      v22 = *i;
      v34.m128i_i64[0] = 0;
      if ( v22 )
      {
        if ( v22 == 1 || v22 == 3 || v22 == 2 )
        {
          v34.m128i_i64[0] = *((_QWORD *)i + 3);
          v13 = 3;
        }
        else
        {
          if ( v22 != 7 )
            BUG();
          v2 = *((unsigned int *)i + 2) | (unsigned __int64)((__int64)*((int *)i + 8) << 32);
          v34.m128i_i32[0] = *((_DWORD *)i + 6);
          v13 = 4;
        }
      }
      else
      {
        v34.m128i_i64[0] = *((unsigned int *)i + 2);
        v13 = 1;
      }
      v14 = *(unsigned int *)(a1 + 72);
      v15 = *(_QWORD *)(a1 + 64);
      v36 = v13;
      v34.m128i_i64[1] = v2;
      v16 = _mm_loadu_si128(&v34);
      v38 = v35;
      v37 = v16;
      v17 = (__m128i *)sub_349EDA0(v15, v15 + 32 * v14, (__int64)&v36);
      if ( v21 == v17 )
        break;
      i += 40;
      result = sub_B0D640(*(_QWORD **)(a1 + 40), v20, (unsigned int)((__int64)((__int64)v17->m128i_i64 - v15) >> 5));
      *(_QWORD *)(a1 + 40) = result;
      if ( (char *)v33 == i )
        return result;
    }
    v24 = v20 + 1;
    v25 = (const __m128i *)&v36;
    if ( v20 + 1 > *(unsigned int *)(a1 + 76) )
    {
      v29 = a1 + 64;
      if ( v15 > (unsigned __int64)&v36 || v17 <= (__m128i *)&v36 )
      {
        sub_C8D5F0(v29, v31, v24, 0x20u, v18, v19);
        v25 = (const __m128i *)&v36;
        v17 = (__m128i *)(*(_QWORD *)(a1 + 64) + 32LL * *(unsigned int *)(a1 + 72));
      }
      else
      {
        sub_C8D5F0(v29, v31, v24, 0x20u, v18, v19);
        v25 = (const __m128i *)((char *)&v36 + *(_QWORD *)(a1 + 64) - v15);
        v17 = (__m128i *)(*(_QWORD *)(a1 + 64) + 32LL * *(unsigned int *)(a1 + 72));
      }
    }
    *v17 = _mm_loadu_si128(v25);
    v17[1] = _mm_loadu_si128(v25 + 1);
    ++*(_DWORD *)(a1 + 72);
    if ( *(_WORD *)(a2 + 68) == 14 )
      v26 = *(_QWORD *)(a2 + 32);
    else
      v26 = *(_QWORD *)(a2 + 32) + 80LL;
    v27 = (__int64)&i[-v26];
    v28 = *(unsigned int *)(a1 + 344);
    result = 0xCCCCCCCCCCCCCCCDLL * (v27 >> 3);
    if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 348) )
    {
      v32 = result;
      sub_C8D5F0(a1 + 336, v30, v28 + 1, 4u, v28 + 1, v19);
      v28 = *(unsigned int *)(a1 + 344);
      result = v32;
    }
    i += 40;
    *(_DWORD *)(*(_QWORD *)(a1 + 336) + 4 * v28) = result;
  }
  return result;
}
