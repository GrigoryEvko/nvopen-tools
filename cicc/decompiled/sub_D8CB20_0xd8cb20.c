// Function: sub_D8CB20
// Address: 0xd8cb20
//
__int64 __fastcall sub_D8CB20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  unsigned __int8 *v10; // rax
  size_t v11; // rdx
  void *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  size_t v16; // [rsp+8h] [rbp-38h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x2Bu )
  {
    v7 = sub_CB6200(*a2, "'Stack Safety Local Analysis' for function '", 0x2Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F72D00);
    qmemcpy(&v8[2], "r function '", 12);
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_3F72D10);
    *(_QWORD *)(v7 + 32) += 44LL;
  }
  v10 = (unsigned __int8 *)sub_BD5D20(a3);
  v12 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v12 < v11 )
  {
    v7 = sub_CB6200(v7, v10, v11);
  }
  else if ( v11 )
  {
    v16 = v11;
    memcpy(v12, v10, v11);
    *(_QWORD *)(v7 + 32) += v16;
  }
  sub_904010(v7, "'\n");
  v13 = sub_BC1CD0(a4, &unk_4F87F28, a3);
  sub_D8CAA0((__int64 *)(v13 + 8), *a2, v14);
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
