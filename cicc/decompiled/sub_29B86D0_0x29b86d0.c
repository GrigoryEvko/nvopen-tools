// Function: sub_29B86D0
// Address: 0x29b86d0
//
void __fastcall sub_29B86D0(unsigned __int64 *a1, __int64 *a2)
{
  unsigned __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r13
  unsigned __int64 v19; // rbx
  __m128i v20; // xmm1
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r13
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  if ( v3 != a1[2] )
  {
    if ( v3 )
    {
      v4 = *a2;
      *(_QWORD *)v3 = *(_QWORD *)(*(_QWORD *)*a2 + 32LL);
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 8) + 32LL);
      *(_QWORD *)(v3 + 16) = 0;
      *(_QWORD *)(v3 + 24) = 0;
      *(_QWORD *)(v3 + 8) = v5;
      *(_QWORD *)(v3 + 32) = 0;
      v6 = (_QWORD *)sub_22077B0(8u);
      *(_QWORD *)(v3 + 16) = v6;
      *(_QWORD *)(v3 + 32) = v6 + 1;
      *v6 = v4;
      *(_QWORD *)(v3 + 24) = v6 + 1;
      *(_QWORD *)(v3 + 48) = 0;
      *(_DWORD *)(v3 + 56) = 0;
      *(_QWORD *)(v3 + 72) = 0;
      *(_DWORD *)(v3 + 80) = 0;
      *(_QWORD *)(v3 + 96) = 0;
      *(_DWORD *)(v3 + 104) = 0;
      *(_WORD *)(v3 + 112) = 0;
      *(_QWORD *)(v3 + 40) = 0xBFF0000000000000LL;
      *(_QWORD *)(v3 + 64) = 0xBFF0000000000000LL;
      *(_QWORD *)(v3 + 88) = 0xBFF0000000000000LL;
      v3 = a1[1];
    }
    a1[1] = v3 + 120;
    return;
  }
  v7 = *a1;
  v8 = v3 - *a1;
  v9 = 0xEEEEEEEEEEEEEEEFLL * (v8 >> 3);
  if ( v9 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)(v3 - v7) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x1111111111111111LL * ((__int64)(v3 - v7) >> 3);
  if ( v11 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_28:
    v26 = sub_22077B0(v22);
    v13 = v26 + 120;
    v25 = v26 + v22;
    goto LABEL_11;
  }
  if ( v12 )
  {
    if ( v12 > 0x111111111111111LL )
      v12 = 0x111111111111111LL;
    v22 = 120 * v12;
    goto LABEL_28;
  }
  v25 = 0;
  v13 = 120;
  v26 = 0;
LABEL_11:
  v14 = v26 + v8;
  if ( v14 )
  {
    v15 = *a2;
    v23 = v13;
    *(_QWORD *)v14 = *(_QWORD *)(*(_QWORD *)*a2 + 32LL);
    v16 = *(_QWORD *)(*(_QWORD *)(v15 + 8) + 32LL);
    *(_QWORD *)(v14 + 16) = 0;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 8) = v16;
    *(_QWORD *)(v14 + 32) = 0;
    v17 = (_QWORD *)sub_22077B0(8u);
    *(_DWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 16) = v17;
    v13 = v23;
    *v17 = v15;
    *(_QWORD *)(v14 + 32) = v17 + 1;
    *(_QWORD *)(v14 + 24) = v17 + 1;
    *(_QWORD *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 72) = 0;
    *(_DWORD *)(v14 + 80) = 0;
    *(_QWORD *)(v14 + 96) = 0;
    *(_DWORD *)(v14 + 104) = 0;
    *(_WORD *)(v14 + 112) = 0;
    *(_QWORD *)(v14 + 40) = 0xBFF0000000000000LL;
    *(_QWORD *)(v14 + 64) = 0xBFF0000000000000LL;
    *(_QWORD *)(v14 + 88) = 0xBFF0000000000000LL;
  }
  if ( v3 != v7 )
  {
    v18 = v26;
    v19 = v7;
    while ( 1 )
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = *(_QWORD *)v19;
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(v18 + 24) = *(_QWORD *)(v19 + 24);
        *(_QWORD *)(v18 + 32) = *(_QWORD *)(v19 + 32);
        v20 = _mm_loadu_si128((const __m128i *)(v19 + 40));
        *(_QWORD *)(v19 + 32) = 0;
        *(_QWORD *)(v19 + 24) = 0;
        *(_QWORD *)(v19 + 16) = 0;
        *(__m128i *)(v18 + 40) = v20;
        *(_QWORD *)(v18 + 56) = *(_QWORD *)(v19 + 56);
        *(__m128i *)(v18 + 64) = _mm_loadu_si128((const __m128i *)(v19 + 64));
        *(_QWORD *)(v18 + 80) = *(_QWORD *)(v19 + 80);
        *(__m128i *)(v18 + 88) = _mm_loadu_si128((const __m128i *)(v19 + 88));
        *(_QWORD *)(v18 + 104) = *(_QWORD *)(v19 + 104);
        *(_BYTE *)(v18 + 112) = *(_BYTE *)(v19 + 112);
        *(_BYTE *)(v18 + 113) = *(_BYTE *)(v19 + 113);
      }
      v21 = *(_QWORD *)(v19 + 16);
      if ( v21 )
        j_j___libc_free_0(v21);
      v19 += 120LL;
      if ( v3 == v19 )
        break;
      v18 += 120;
    }
    v13 = v18 + 240;
  }
  if ( v7 )
  {
    v24 = v13;
    j_j___libc_free_0(v7);
    v13 = v24;
  }
  a1[1] = v13;
  *a1 = v26;
  a1[2] = v25;
}
