// Function: sub_6EFFF0
// Address: 0x6efff0
//
__int64 __fastcall sub_6EFFF0(__int64 a1, _QWORD *a2, __int64 a3, int a4, const __m128i *a5, __int64 *a6, _DWORD *a7)
{
  __int64 v7; // r12
  bool v8; // di
  int v9; // ebx
  __int64 v11; // r14
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i v20; // xmm0
  _QWORD *v22; // [rsp+8h] [rbp-78h]
  int v23; // [rsp+10h] [rbp-70h]
  unsigned int v24; // [rsp+14h] [rbp-6Ch]
  unsigned int v25; // [rsp+18h] [rbp-68h]
  int v28; // [rsp+30h] [rbp-50h]
  _QWORD *v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h] BYREF
  __int64 v31[7]; // [rsp+48h] [rbp-38h] BYREF

  v7 = a1;
  v28 = a1 + 28;
  v30 = 0;
  v29 = a2;
  v8 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 4) != 0;
  v9 = a2[2] - 1;
  if ( v9 < 0 )
  {
    v16 = 0;
    goto LABEL_17;
  }
  v25 = a4 | 0x4004;
  v11 = 24LL * v9;
  v24 = a4 | 4;
  while ( !*a7 )
  {
    v12 = v11 + *v29;
    if ( v9 )
    {
      v13 = *(_BYTE *)(v12 + 16);
      v14 = a4 | 0x6004u;
      if ( (v13 & 4) == 0 )
        v14 = v25;
      v15 = (unsigned int)v14 | 0x80000;
      if ( (v13 & 8) != 0 )
        v14 = (unsigned int)v15;
      if ( !v8 )
      {
        a2 = qword_4F04C68;
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) |= 4u;
      }
    }
    else
    {
      a2 = qword_4F04C68;
      v14 = v24;
      v18 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v15 = *(_BYTE *)(v18 + 6) & 0xFB;
      LOBYTE(v15) = (4 * v8) | *(_BYTE *)(v18 + 6) & 0xFB;
      *(_BYTE *)(v18 + 6) = v15;
    }
    if ( v7 )
    {
      a2 = *(_QWORD **)(v12 + 8);
      v7 = sub_7410C0(v7, (_DWORD)a2, *(_QWORD *)v12, 0, v28, v14, (__int64)a7, a3, (__int64)a5, (__int64)&v30);
LABEL_4:
      --v9;
      v11 -= 24;
      if ( v9 == -1 )
        break;
    }
    else
    {
      if ( !v30 )
      {
        v22 = (_QWORD *)v12;
        v23 = v14;
        v19 = sub_724DC0(0, a2, v12, v15, a5, v14);
        v20 = _mm_loadu_si128(a5);
        v31[0] = v19;
        *(__m128i *)v19 = v20;
        *(__m128i *)(v19 + 16) = _mm_loadu_si128(a5 + 1);
        *(__m128i *)(v19 + 32) = _mm_loadu_si128(a5 + 2);
        *(__m128i *)(v19 + 48) = _mm_loadu_si128(a5 + 3);
        *(__m128i *)(v19 + 64) = _mm_loadu_si128(a5 + 4);
        *(__m128i *)(v19 + 80) = _mm_loadu_si128(a5 + 5);
        *(__m128i *)(v19 + 96) = _mm_loadu_si128(a5 + 6);
        *(__m128i *)(v19 + 112) = _mm_loadu_si128(a5 + 7);
        *(__m128i *)(v19 + 128) = _mm_loadu_si128(a5 + 8);
        *(__m128i *)(v19 + 144) = _mm_loadu_si128(a5 + 9);
        *(__m128i *)(v19 + 160) = _mm_loadu_si128(a5 + 10);
        *(__m128i *)(v19 + 176) = _mm_loadu_si128(a5 + 11);
        *(__m128i *)(v19 + 192) = _mm_loadu_si128(a5 + 12);
        a2 = (_QWORD *)v22[1];
        v30 = sub_743600(v31[0], (_DWORD)a2, *v22, 0, v28, v23, (__int64)a7, a3, (__int64)a5);
        sub_724E30(v31);
        goto LABEL_4;
      }
      a2 = *(_QWORD **)(v12 + 8);
      --v9;
      v11 -= 24;
      v30 = sub_743600(v30, (_DWORD)a2, *(_QWORD *)v12, 0, v28, v14, (__int64)a7, a3, (__int64)a5);
      if ( v9 == -1 )
        break;
    }
  }
  v16 = v30;
LABEL_17:
  *a6 = v16;
  return v7;
}
