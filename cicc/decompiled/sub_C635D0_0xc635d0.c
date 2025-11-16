// Function: sub_C635D0
// Address: 0xc635d0
//
char *__fastcall sub_C635D0(char *a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __m128i v5; // xmm0
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __m128i si128; // xmm0
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD v15[2]; // [rsp+8h] [rbp-18h] BYREF

  switch ( a3 )
  {
    case 2:
      v15[0] = 22;
      *(_QWORD *)a1 = a1 + 16;
      v10 = sub_22409D0(a1, v15, 0);
      v11 = v15[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F668A0);
      *(_QWORD *)a1 = v10;
      *((_QWORD *)a1 + 2) = v11;
      *(_DWORD *)(v10 + 16) = 1701999221;
      *(_WORD *)(v10 + 20) = 11876;
      *(__m128i *)v10 = si128;
      v13 = v15[0];
      v14 = *(_QWORD *)a1;
      *((_QWORD *)a1 + 1) = v15[0];
      *(_BYTE *)(v14 + v13) = 0;
      return a1;
    case 3:
      v15[0] = 123;
      *(_QWORD *)a1 = a1 + 16;
      v3 = sub_22409D0(a1, v15, 0);
      v4 = v15[0];
      v5 = _mm_load_si128((const __m128i *)&xmmword_3F66830);
      *(_QWORD *)a1 = v3;
      *((_QWORD *)a1 + 2) = v4;
      *(__m128i *)v3 = v5;
      v6 = _mm_load_si128((const __m128i *)&xmmword_3F66840);
      qmemcpy((void *)(v3 + 112), "file a bug.", 11);
      *(__m128i *)(v3 + 16) = v6;
      *(__m128i *)(v3 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F66850);
      *(__m128i *)(v3 + 48) = _mm_load_si128((const __m128i *)&xmmword_3F66860);
      *(__m128i *)(v3 + 64) = _mm_load_si128((const __m128i *)&xmmword_3F66870);
      *(__m128i *)(v3 + 80) = _mm_load_si128((const __m128i *)&xmmword_3F66880);
      *(__m128i *)(v3 + 96) = _mm_load_si128((const __m128i *)&xmmword_3F66890);
      v7 = v15[0];
      v8 = *(_QWORD *)a1;
      *((_QWORD *)a1 + 1) = v15[0];
      *(_BYTE *)(v8 + v7) = 0;
      return a1;
    case 1:
      *(_QWORD *)a1 = a1 + 16;
      strcpy(a1 + 16, "Multiple errors");
      *((_QWORD *)a1 + 1) = 15;
      return a1;
    default:
      BUG();
  }
}
