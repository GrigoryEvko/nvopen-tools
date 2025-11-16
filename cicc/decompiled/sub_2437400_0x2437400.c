// Function: sub_2437400
// Address: 0x2437400
//
unsigned __int64 __fastcall sub_2437400(__int64 a1, __int64 *a2, unsigned int *a3, _BYTE *a4, __int64 *a5)
{
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  char v17; // dl
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  const __m128i *v21; // r12
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r8
  const void *v24; // rsi
  char *v25; // r12
  unsigned __int64 v26; // [rsp+10h] [rbp-80h] BYREF
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h]
  unsigned __int64 v29; // [rsp+28h] [rbp-68h]
  char v30; // [rsp+30h] [rbp-60h]
  char v31; // [rsp+39h] [rbp-57h]
  __int64 v32; // [rsp+40h] [rbp-50h]
  __int64 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]

  v5 = *a2;
  v6 = *(_DWORD *)(a1 + 8);
  v7 = *a3;
  v8 = *a5;
  if ( *(_DWORD *)(a1 + 12) <= v6 )
  {
    LOBYTE(v27) = *a4;
    v28 = v8;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v15 = sub_B43CC0(v5);
    v29 = (sub_9208B0(v15, v8) + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v30 = v17;
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
      v18 = *(_QWORD *)(v5 - 8);
    else
      v18 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
    v19 = *(unsigned int *)(a1 + 8);
    v20 = *(unsigned int *)(a1 + 12);
    v21 = (const __m128i *)&v26;
    v22 = *(_QWORD *)a1;
    v23 = v19 + 1;
    v26 = 32 * v7 + v18;
    if ( v19 + 1 > v20 )
    {
      v24 = (const void *)(a1 + 16);
      if ( v22 > (unsigned __int64)&v26 || (unsigned __int64)&v26 >= v22 + 72 * v19 )
      {
        sub_C8D5F0(a1, v24, v23, 0x48u, v23, v16);
        v22 = *(_QWORD *)a1;
        v19 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        v25 = (char *)&v26 - v22;
        sub_C8D5F0(a1, v24, v23, 0x48u, v23, v16);
        v22 = *(_QWORD *)a1;
        v19 = *(unsigned int *)(a1 + 8);
        v21 = (const __m128i *)&v25[*(_QWORD *)a1];
      }
    }
    result = v22 + 72 * v19;
    *(__m128i *)result = _mm_loadu_si128(v21);
    *(__m128i *)(result + 16) = _mm_loadu_si128(v21 + 1);
    *(__m128i *)(result + 32) = _mm_loadu_si128(v21 + 2);
    *(__m128i *)(result + 48) = _mm_loadu_si128(v21 + 3);
    *(_QWORD *)(result + 64) = v21[4].m128i_i64[0];
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v9 = *(_QWORD *)a1 + 72LL * v6;
    if ( v9 )
    {
      *(_BYTE *)(v9 + 8) = *a4;
      *(_QWORD *)(v9 + 16) = v8;
      *(_QWORD *)(v9 + 24) = 0;
      *(_BYTE *)(v9 + 32) = 0;
      *(_BYTE *)(v9 + 41) = 0;
      *(_QWORD *)(v9 + 48) = 0;
      *(_QWORD *)(v9 + 56) = 0;
      *(_QWORD *)(v9 + 64) = 0;
      v10 = sub_B43CC0(v5);
      v11 = sub_9208B0(v10, v8);
      v27 = v12;
      v26 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v9 + 24) = v26;
      *(_BYTE *)(v9 + 32) = v27;
      if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
        v13 = *(_QWORD *)(v5 - 8);
      else
        v13 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
      *(_QWORD *)v9 = 32 * v7 + v13;
      v6 = *(_DWORD *)(a1 + 8);
    }
    result = v6 + 1;
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
