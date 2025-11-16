// Function: sub_E59780
// Address: 0xe59780
//
_BYTE *__fastcall sub_E59780(__int64 a1, signed __int64 a2, signed __int64 a3, signed __int64 a4)
{
  __int64 v7; // rdi
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // rdi
  _WORD *v11; // rdx
  __int64 v12; // rdi
  _WORD *v13; // rdx

  sub_E9CE20();
  v7 = *(_QWORD *)(a1 + 304);
  v8 = *(__m128i **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 0x19u )
  {
    sub_CB6200(v7, "\t.cfi_llvm_def_aspace_cfa ", 0x1Au);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7FA00);
    qmemcpy(&v8[1], "space_cfa ", 10);
    *v8 = si128;
    *(_QWORD *)(v7 + 32) += 26LL;
  }
  sub_E4C9A0(a1, a2);
  v10 = *(_QWORD *)(a1 + 304);
  v11 = *(_WORD **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
  {
    v10 = sub_CB6200(v10, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v11 = 8236;
    *(_QWORD *)(v10 + 32) += 2LL;
  }
  sub_CB59F0(v10, a3);
  v12 = *(_QWORD *)(a1 + 304);
  v13 = *(_WORD **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
  {
    v12 = sub_CB6200(v12, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v13 = 8236;
    *(_QWORD *)(v12 + 32) += 2LL;
  }
  sub_CB59F0(v12, a4);
  return sub_E4D880(a1);
}
