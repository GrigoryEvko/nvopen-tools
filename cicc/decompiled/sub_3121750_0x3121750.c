// Function: sub_3121750
// Address: 0x3121750
//
__int64 __fastcall sub_3121750(__int64 a1, int *a2)
{
  int v2; // ebx
  _QWORD *v3; // rax
  void *v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  __m128i *v7; // rdx
  __int64 v8; // rdi
  __m128i si128; // xmm0

  v2 = *a2;
  v3 = sub_CB72A0();
  v4 = (void *)v3[4];
  v5 = (__int64)v3;
  if ( v3[3] - (_QWORD)v4 <= 0xEu )
  {
    v5 = sub_CB6200((__int64)v3, "Error of kind: ", 0xFu);
  }
  else
  {
    qmemcpy(v4, "Error of kind: ", 15);
    v3[4] += 15LL;
  }
  v6 = sub_CB59F0(v5, v2);
  v7 = *(__m128i **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0x4Du )
    return sub_CB6200(v6, " when emitting offload entries and metadata during OMPIRBuilder finalization \n", 0x4Eu);
  si128 = _mm_load_si128((const __m128i *)&xmmword_44D02C0);
  qmemcpy(&v7[4], "finalization \n", 14);
  *v7 = si128;
  v7[1] = _mm_load_si128((const __m128i *)&xmmword_44D02D0);
  v7[2] = _mm_load_si128((const __m128i *)&xmmword_44D02E0);
  v7[3] = _mm_load_si128((const __m128i *)&xmmword_44D02F0);
  *(_QWORD *)(v8 + 32) += 78LL;
  return 2592;
}
