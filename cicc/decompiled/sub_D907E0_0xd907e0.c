// Function: sub_D907E0
// Address: 0xd907e0
//
__int64 __fastcall sub_D907E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  void *v11; // rdi
  size_t v12; // rdx
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x23u )
  {
    v16 = sub_CB6200(*a2, "'Stack Safety Analysis' for module '", 0x24u);
    v11 = *(void **)(v16 + 32);
    v8 = v16;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F72D20);
    v9[2].m128i_i32[0] = 656434540;
    *v9 = si128;
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_3F72D30);
    v11 = (void *)(*(_QWORD *)(v8 + 32) + 36LL);
    *(_QWORD *)(v8 + 32) = v11;
  }
  v12 = *(_QWORD *)(a3 + 176);
  v13 = *(unsigned __int8 **)(a3 + 168);
  if ( v12 > *(_QWORD *)(v8 + 24) - (_QWORD)v11 )
  {
    v8 = sub_CB6200(v8, v13, v12);
  }
  else if ( v12 )
  {
    v17 = *(_QWORD *)(a3 + 176);
    memcpy(v11, v13, v12);
    *(_QWORD *)(v8 + 32) += v17;
  }
  sub_904010(v8, "'\n");
  v14 = sub_BC0510(a4, &unk_4F87F18, a3);
  sub_D90530((__int64 *)(v14 + 8), *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
