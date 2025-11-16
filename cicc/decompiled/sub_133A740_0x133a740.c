// Function: sub_133A740
// Address: 0x133a740
//
__int64 __fastcall sub_133A740(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // r13
  __int64 v3; // r12
  void *v4; // rsp
  unsigned int v5; // eax
  __int64 v6; // rdx
  unsigned int v7; // edx
  unsigned int v8; // r15d
  __int64 v9; // r13
  _BYTE *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // r10
  _BYTE *v13; // rdi
  _QWORD *v14; // r14
  const __m128i *v15; // rbx
  __int64 v16; // rdi
  __m128i *v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  __int64 v22; // rax
  __m128i si128; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __int64 result; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

  v28 = a1;
  v1 = sub_1322320(4096);
  v2 = qword_4F96BA0;
  v3 = (__int64)v1;
  v4 = alloca(16 * ((8 * (unsigned __int64)*(unsigned int *)(qword_4F96BA0 + 8) + 15) >> 4));
  sub_131DCA0((__int64)v1);
  if ( *(_DWORD *)(v2 + 8) )
  {
    v5 = 0;
    do
    {
      v6 = v5++;
      *(&v28 + v6) = qword_50579C0[v6];
      v7 = *(_DWORD *)(qword_4F96BA0 + 8);
    }
    while ( v7 > v5 );
    if ( v7 )
    {
      v8 = 0;
      v9 = __readfsqword(0) - 2664;
      do
      {
        v10 = (_BYTE *)v9;
        if ( __readfsbyte(0xFFFFF8C8) )
          v10 = (_BYTE *)sub_1313D30(v9, 0);
        v11 = sub_1322110(v10, v8, 1, 0);
        v12 = *(&v28 + v8);
        *((_BYTE *)v11 + 4) = v12 != 0;
        if ( v12 )
        {
          v13 = (_BYTE *)v9;
          if ( __readfsbyte(0xFFFFF8C8) )
          {
            v29 = v12;
            v27 = sub_1313D30(v9, 0);
            v12 = v29;
            v13 = (_BYTE *)v27;
          }
          v29 = v12;
          v14 = sub_1322110(v13, v8, 1, 0);
          sub_131DCA0((__int64)v14);
          sub_131DE10(v28, (__int64)v14, v29);
          sub_13226C0(v3, (__int64)v14, 0);
        }
        ++v8;
      }
      while ( *(_DWORD *)(qword_4F96BA0 + 8) > v8 );
    }
  }
  v15 = (const __m128i *)qword_4F96BA8;
  *(_QWORD *)qword_4F96BA8 = *(_QWORD *)(*(_QWORD *)(v3 + 80) + 10368LL) + *(_QWORD *)(*(_QWORD *)(v3 + 80) + 40LL);
  v15->m128i_i64[1] = *(_QWORD *)(v3 + 56) << 12;
  v15[1].m128i_i64[0] = **(_QWORD **)(v3 + 80) + *(_QWORD *)(*(_QWORD *)(v3 + 80) + 32LL);
  v16 = v28;
  v15[2].m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v3 + 80) + 8LL);
  v15[1].m128i_i64[1] = *(_QWORD *)(*(_QWORD *)(v3 + 80) + 16LL);
  v15[2].m128i_i64[1] = *(_QWORD *)(*(_QWORD *)(v3 + 80) + 24LL);
  v15[3].m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v3 + 80) + 144LL);
  if ( (unsigned __int8)sub_131AB20(v16, (__int64)&v15[3].m128i_i64[1]) )
  {
    v15[3].m128i_i64[1] = 0;
    v15[8].m128i_i64[1] = 0;
    memset(
      (void *)((unsigned __int64)&v15[4] & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((_DWORD)v15 + 56 - (((_DWORD)v15 + 64) & 0xFFFFFFF8) + 88) >> 3));
    sub_130B140(&v15[4].m128i_i64[1], qword_4287888);
  }
  v17 = (__m128i *)qword_4F96BA8;
  *(__m128i *)(qword_4F96BA8 + 208) = _mm_loadu_si128(v15 + 5);
  v17[14] = _mm_loadu_si128(v15 + 6);
  v17[15] = _mm_loadu_si128(v15 + 7);
  v17[16] = _mm_loadu_si128(v15 + 8);
  v17[15].m128i_i32[1] = 0;
  if ( pthread_mutex_trylock(&stru_5260DA0) )
  {
    sub_130AD90((__int64)qword_5260D60);
    unk_5260DC8 = 1;
  }
  ++unk_5260D98;
  if ( v28 != unk_5260D90 )
  {
    ++unk_5260D88;
    unk_5260D90 = v28;
  }
  v18 = qword_4F96BA8;
  v19 = _mm_loadu_si128((const __m128i *)&qword_5260D60[2]);
  v20 = _mm_loadu_si128((const __m128i *)&qword_5260D60[4]);
  v21 = _mm_loadu_si128((const __m128i *)&unk_5260D90);
  *(__m128i *)(qword_4F96BA8 + 144) = _mm_loadu_si128((const __m128i *)qword_5260D60);
  *(__m128i *)(v18 + 160) = v19;
  *(__m128i *)(v18 + 176) = v20;
  *(__m128i *)(v18 + 192) = v21;
  *(_DWORD *)(v18 + 180) = 0;
  unk_5260DC8 = 0;
  pthread_mutex_unlock(&stru_5260DA0);
  v22 = qword_4F96BA8;
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F96BD0);
  v24 = _mm_load_si128((const __m128i *)&unk_4F96BE0);
  v25 = _mm_load_si128((const __m128i *)&xmmword_4F96BF0);
  *(__m128i *)(qword_4F96BA8 + 272) = _mm_load_si128((const __m128i *)&xmmword_4F96BC0);
  *(__m128i *)(v22 + 288) = si128;
  *(__m128i *)(v22 + 304) = v24;
  *(__m128i *)(v22 + 320) = v25;
  *(_DWORD *)(v22 + 308) = 0;
  result = qword_4F96BA0;
  ++*(_QWORD *)qword_4F96BA0;
  return result;
}
