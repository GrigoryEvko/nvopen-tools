// Function: sub_2437640
// Address: 0x2437640
//
unsigned __int64 __fastcall sub_2437640(__int64 a1, __int64 *a2, unsigned int *a3, _BYTE *a4, __int64 *a5, char *a6)
{
  unsigned int v6; // eax
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r15
  char v10; // si
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  unsigned __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // r9
  char v19; // dl
  __int64 v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  const __m128i *v23; // r12
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // r8
  const void *v26; // rsi
  char *v27; // r12
  unsigned __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  __int64 v29; // [rsp+18h] [rbp-78h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  unsigned __int64 v31; // [rsp+28h] [rbp-68h]
  char v32; // [rsp+30h] [rbp-60h]
  char v33; // [rsp+38h] [rbp-58h]
  char v34; // [rsp+39h] [rbp-57h]
  __int64 v35; // [rsp+40h] [rbp-50h]
  __int64 v36; // [rsp+48h] [rbp-48h]
  __int64 v37; // [rsp+50h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 8);
  v7 = *a2;
  v8 = *a3;
  v9 = *a5;
  v10 = *a6;
  if ( *(_DWORD *)(a1 + 12) <= v6 )
  {
    LOBYTE(v29) = *a4;
    v33 = v10;
    v30 = v9;
    v34 = 1;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v17 = sub_B43CC0(v7);
    v31 = (sub_9208B0(v17, v9) + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v32 = v19;
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      v20 = *(_QWORD *)(v7 - 8);
    else
      v20 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    v21 = *(unsigned int *)(a1 + 8);
    v22 = *(unsigned int *)(a1 + 12);
    v23 = (const __m128i *)&v28;
    v24 = *(_QWORD *)a1;
    v25 = v21 + 1;
    v28 = 32 * v8 + v20;
    if ( v21 + 1 > v22 )
    {
      v26 = (const void *)(a1 + 16);
      if ( v24 > (unsigned __int64)&v28 || (unsigned __int64)&v28 >= v24 + 72 * v21 )
      {
        sub_C8D5F0(a1, v26, v25, 0x48u, v25, v18);
        v24 = *(_QWORD *)a1;
        v21 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        v27 = (char *)&v28 - v24;
        sub_C8D5F0(a1, v26, v25, 0x48u, v25, v18);
        v24 = *(_QWORD *)a1;
        v21 = *(unsigned int *)(a1 + 8);
        v23 = (const __m128i *)&v27[*(_QWORD *)a1];
      }
    }
    result = v24 + 72 * v21;
    *(__m128i *)result = _mm_loadu_si128(v23);
    *(__m128i *)(result + 16) = _mm_loadu_si128(v23 + 1);
    *(__m128i *)(result + 32) = _mm_loadu_si128(v23 + 2);
    *(__m128i *)(result + 48) = _mm_loadu_si128(v23 + 3);
    *(_QWORD *)(result + 64) = v23[4].m128i_i64[0];
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v11 = *(_QWORD *)a1 + 72LL * v6;
    if ( v11 )
    {
      *(_BYTE *)(v11 + 8) = *a4;
      *(_BYTE *)(v11 + 40) = v10;
      *(_QWORD *)(v11 + 16) = v9;
      *(_QWORD *)(v11 + 24) = 0;
      *(_BYTE *)(v11 + 32) = 0;
      *(_BYTE *)(v11 + 41) = 1;
      *(_QWORD *)(v11 + 48) = 0;
      *(_QWORD *)(v11 + 56) = 0;
      *(_QWORD *)(v11 + 64) = 0;
      v12 = sub_B43CC0(v7);
      v13 = sub_9208B0(v12, v9);
      v29 = v14;
      v28 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v11 + 24) = v28;
      *(_BYTE *)(v11 + 32) = v29;
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
        v15 = *(_QWORD *)(v7 - 8);
      else
        v15 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      *(_QWORD *)v11 = 32 * v8 + v15;
      v6 = *(_DWORD *)(a1 + 8);
    }
    result = v6 + 1;
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
