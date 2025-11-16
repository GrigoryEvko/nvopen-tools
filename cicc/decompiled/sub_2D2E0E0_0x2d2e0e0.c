// Function: sub_2D2E0E0
// Address: 0x2d2e0e0
//
__int64 __fastcall sub_2D2E0E0(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  int v10; // eax
  __int64 v11; // r8
  int v12; // r15d
  unsigned int v13; // ecx
  __m128i *v14; // rdi
  __m128i *v15; // rdx
  __int32 v16; // r10d
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 i; // rax
  __int64 j; // rax
  __m128i *v25; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    v25 = 0;
    *(_QWORD *)a2 = v9 + 1;
LABEL_26:
    LODWORD(v8) = 2 * v8;
    goto LABEL_27;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 1;
  v13 = (v8 - 1) & (37 * *a3);
  v14 = (__m128i *)(v11 + 216LL * v13);
  v15 = 0;
  v16 = v14->m128i_i32[0];
  if ( v10 == v14->m128i_i32[0] )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v14;
    *(_QWORD *)(a1 + 24) = v11 + 216 * v8;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( v16 != -1 )
  {
    if ( !v15 && v16 == -2 )
      v15 = v14;
    v13 = (v8 - 1) & (v12 + v13);
    v14 = (__m128i *)(v11 + 216LL * v13);
    v16 = v14->m128i_i32[0];
    if ( v10 == v14->m128i_i32[0] )
      goto LABEL_3;
    ++v12;
  }
  v18 = *(_DWORD *)(a2 + 16);
  if ( !v15 )
    v15 = v14;
  v19 = v18 + 1;
  *(_QWORD *)a2 = v9 + 1;
  v25 = v15;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v8) )
    goto LABEL_26;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v8 >> 3 )
  {
LABEL_27:
    sub_2D2DD70(a2, v8);
    sub_2D28C60(a2, a3, &v25);
    v15 = v25;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( v15->m128i_i32[0] != -1 )
    --*(_DWORD *)(a2 + 20);
  v15->m128i_i32[0] = *a3;
  v20 = *(_QWORD *)(a4 + 200);
  v15[12].m128i_i64[1] = 0;
  v15[13].m128i_i64[0] = v20;
  v15[12].m128i_i32[2] = *(_DWORD *)(a4 + 192);
  v15[12].m128i_i32[3] = *(_DWORD *)(a4 + 196);
  v15[13].m128i_i64[0] = *(_QWORD *)(a4 + 200);
  if ( *(_DWORD *)(a4 + 192) )
  {
    v15[1] = _mm_loadu_si128((const __m128i *)(a4 + 8));
    v15[2] = _mm_loadu_si128((const __m128i *)(a4 + 24));
    v15[3] = _mm_loadu_si128((const __m128i *)(a4 + 40));
    v15[4] = _mm_loadu_si128((const __m128i *)(a4 + 56));
    v15[5] = _mm_loadu_si128((const __m128i *)(a4 + 72));
    v15[6] = _mm_loadu_si128((const __m128i *)(a4 + 88));
    v15[7] = _mm_loadu_si128((const __m128i *)(a4 + 104));
    v15[8] = _mm_loadu_si128((const __m128i *)(a4 + 120));
    v15[9] = _mm_loadu_si128((const __m128i *)(a4 + 136));
    v15[10] = _mm_loadu_si128((const __m128i *)(a4 + 152));
    v15[11] = _mm_loadu_si128((const __m128i *)(a4 + 168));
    v15[12].m128i_i32[0] = *(_DWORD *)(a4 + 184);
    *(_DWORD *)(a4 + 192) = 0;
    *(_QWORD *)a4 = 0;
    *(_QWORD *)(a4 + 184) = 0;
    memset(
      (void *)((a4 + 8) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)a4 - (((_DWORD)a4 + 8) & 0xFFFFFFF8) + 192) >> 3));
  }
  else
  {
    for ( i = 0; i != 32; i += 2 )
    {
      v15->m128i_i32[i + 2] = *(_DWORD *)(a4 + i * 4);
      v15->m128i_i32[i + 3] = *(_DWORD *)(a4 + i * 4 + 4);
    }
    for ( j = 0; j != 16; ++j )
      v15[8].m128i_i32[j + 2] = *(_DWORD *)(a4 + j * 4 + 128);
  }
  v21 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v15;
  *(_BYTE *)(a1 + 32) = 1;
  v22 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 216 * v21;
  *(_QWORD *)(a1 + 8) = v22;
  return a1;
}
