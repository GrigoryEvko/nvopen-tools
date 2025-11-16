// Function: sub_255FEC0
// Address: 0x255fec0
//
__int64 __fastcall sub_255FEC0(__int64 a1, __int64 **a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 *v6; // r15
  _QWORD *v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rax
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __int64 v21; // rax
  unsigned int v22[13]; // [rsp+Ch] [rbp-34h] BYREF

  v5 = *a3;
  v6 = *a2;
  v7 = (_QWORD *)(*a3 + 72);
  if ( (!(unsigned __int8)sub_A73ED0(v7, 23) && !(unsigned __int8)sub_B49560(v5, 23)
     || (unsigned __int8)sub_A73ED0(v7, 4)
     || (unsigned __int8)sub_B49560(v5, 4))
    && (v8 = *(_QWORD *)(v5 - 32)) != 0
    && !*(_BYTE *)v8
    && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v5 + 80)
    && sub_981210(*v6, v8, v22)
    && v22[0] == 109 )
  {
    sub_B18290(a4, "Moving globalized variable to the stack.", 0x28u);
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a4 + 8);
    *(_BYTE *)(a1 + 12) = *(_BYTE *)(a4 + 12);
    v18 = _mm_loadu_si128((const __m128i *)(a4 + 24));
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a4 + 16);
    *(__m128i *)(a1 + 24) = v18;
    v19 = _mm_loadu_si128((const __m128i *)(a4 + 48));
    v20 = _mm_loadu_si128((const __m128i *)(a4 + 64));
    *(_QWORD *)a1 = &unk_49D9D40;
    v21 = *(_QWORD *)(a4 + 40);
    *(__m128i *)(a1 + 48) = v19;
    *(_QWORD *)(a1 + 40) = v21;
    *(__m128i *)(a1 + 64) = v20;
  }
  else
  {
    sub_B18290(a4, "Moving memory allocation from the heap to the stack.", 0x34u);
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a4 + 8);
    *(_BYTE *)(a1 + 12) = *(_BYTE *)(a4 + 12);
    v13 = _mm_loadu_si128((const __m128i *)(a4 + 24));
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a4 + 16);
    *(__m128i *)(a1 + 24) = v13;
    v14 = _mm_loadu_si128((const __m128i *)(a4 + 48));
    v15 = _mm_loadu_si128((const __m128i *)(a4 + 64));
    *(_QWORD *)a1 = &unk_49D9D40;
    v16 = *(_QWORD *)(a4 + 40);
    *(__m128i *)(a1 + 48) = v14;
    *(_QWORD *)(a1 + 40) = v16;
    *(__m128i *)(a1 + 64) = v15;
  }
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  if ( *(_DWORD *)(a4 + 88) )
    sub_255FC40(a1 + 80, a4 + 80, v9, v10, v11, v12);
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(a4 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(a4 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(a4 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  return a1;
}
