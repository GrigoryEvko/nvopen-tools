// Function: sub_30CC980
// Address: 0x30cc980
//
__int64 __fastcall sub_30CC980(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r11d
  unsigned int i; // eax
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 v20; // rdi
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // r13
  __int64 *v24; // rax
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64, __int64); // rax
  __m128i *v27; // rdx
  __m128i si128; // xmm0
  __int64 v29; // [rsp+10h] [rbp-A0h] BYREF
  char v30; // [rsp+90h] [rbp-20h] BYREF

  v8 = sub_227ED20(a4, &qword_4FDADB8, (__int64 *)a3, a5);
  if ( *(_DWORD *)(a3 + 16) )
  {
    v12 = *(_QWORD *)(v8 + 8);
    v13 = *(_QWORD *)(v12 + 72);
    v14 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a3 + 8) + 8LL) + 40LL);
    v15 = *(unsigned int *)(v12 + 88);
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    v16 = 1;
    for ( i = (v15 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_502F110 >> 9) ^ ((unsigned int)&unk_502F110 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; i = (v15 - 1) & v19 )
    {
      v18 = v13 + 24LL * i;
      if ( *(_UNKNOWN **)v18 == &unk_502F110 && v14 == *(_QWORD *)(v18 + 8) )
        break;
      if ( *(_QWORD *)v18 == -4096 && *(_QWORD *)(v18 + 8) == -4096 )
        goto LABEL_12;
      v19 = v16 + i;
      ++v16;
    }
    if ( v18 != v13 + 24 * v15 && (v23 = *(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL)) != 0 )
    {
      v24 = &v29;
      do
      {
        *v24 = -4096;
        v24 += 2;
      }
      while ( v24 != (__int64 *)&v30 );
      v25 = *a2;
      v26 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)(v23 + 24) + 32LL);
      if ( v26 == sub_30CA850 )
      {
        v27 = *(__m128i **)(v25 + 32);
        if ( *(_QWORD *)(v25 + 24) - (_QWORD)v27 <= 0x21u )
        {
          sub_CB6200(v25, "Unimplemented InlineAdvisor print\n", 0x22u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44CCA90);
          v27[2].m128i_i16[0] = 2676;
          *v27 = si128;
          v27[1] = _mm_load_si128((const __m128i *)&xmmword_44CCAA0);
          *(_QWORD *)(v25 + 32) += 34LL;
        }
      }
      else
      {
        v26(*(_QWORD *)(v23 + 24), *a2);
      }
    }
    else
    {
LABEL_12:
      v20 = *a2;
      v21 = *(__m128i **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v21 <= 0x11u )
      {
        sub_CB6200(v20, "No Inline Advisor\n", 0x12u);
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_44CCAB0);
        v21[1].m128i_i16[0] = 2674;
        *v21 = v22;
        *(_QWORD *)(v20 + 32) += 18LL;
      }
    }
  }
  else
  {
    v9 = *a2;
    v10 = *(void **)(*a2 + 32);
    if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v10 <= 0xDu )
    {
      sub_CB6200(v9, "SCC is empty!\n", 0xEu);
    }
    else
    {
      qmemcpy(v10, "SCC is empty!\n", 14);
      *(_QWORD *)(v9 + 32) += 14LL;
    }
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
