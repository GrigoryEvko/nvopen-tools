// Function: sub_30CC760
// Address: 0x30cc760
//
__int64 __fastcall sub_30CC760(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // ebx
  unsigned int i; // eax
  __int64 v11; // r9
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 (__fastcall *v16)(__int64, __int64); // rax
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __m128i *v20; // rdx
  __m128i si128; // xmm0

  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v7 )
  {
LABEL_16:
    v13 = *a2;
LABEL_17:
    v20 = *(__m128i **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v20 <= 0x11u )
    {
      sub_CB6200(v13, "No Inline Advisor\n", 0x12u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44CCAB0);
      v20[1].m128i_i16[0] = 2674;
      *v20 = si128;
      *(_QWORD *)(v13 + 32) += 18LL;
    }
    goto LABEL_12;
  }
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_502F110 >> 9) ^ ((unsigned int)&unk_502F110 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v11 == &unk_502F110 && a3 == *(_QWORD *)(v11 + 8) )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
      goto LABEL_16;
    v12 = v9 + i;
    ++v9;
  }
  v13 = *a2;
  if ( v11 == v8 + 24 * v7 )
    goto LABEL_17;
  v14 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
  if ( !v14 )
    goto LABEL_17;
  v15 = *(_QWORD *)(v14 + 24);
  v16 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 32LL);
  if ( v16 == sub_30CA850 )
  {
    v17 = *(__m128i **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v17 <= 0x21u )
    {
      sub_CB6200(v13, "Unimplemented InlineAdvisor print\n", 0x22u);
    }
    else
    {
      v18 = _mm_load_si128((const __m128i *)&xmmword_44CCA90);
      v17[2].m128i_i16[0] = 2676;
      *v17 = v18;
      v17[1] = _mm_load_si128((const __m128i *)&xmmword_44CCAA0);
      *(_QWORD *)(v13 + 32) += 34LL;
    }
  }
  else
  {
    v16(v15, v13);
  }
LABEL_12:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
