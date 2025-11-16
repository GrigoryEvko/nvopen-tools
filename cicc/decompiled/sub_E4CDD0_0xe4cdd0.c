// Function: sub_E4CDD0
// Address: 0xe4cdd0
//
_BYTE *__fastcall sub_E4CDD0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rdi
  __m128i *v5; // rdx
  __int64 v6; // rax
  _WORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdi
  _BYTE *result; // rax

  v3 = *(_QWORD *)(a1 + 304);
  v5 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v5 <= 0xFu )
  {
    v3 = sub_CB6200(v3, "\t.gnu_attribute ", 0x10u);
  }
  else
  {
    *v5 = _mm_load_si128((const __m128i *)&xmmword_3F7F820);
    *(_QWORD *)(v3 + 32) += 16LL;
  }
  v6 = sub_CB59D0(v3, a2);
  v7 = *(_WORD **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 1u )
  {
    v8 = sub_CB6200(v6, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v7 = 8236;
    *(_QWORD *)(v6 + 32) += 2LL;
  }
  v9 = sub_CB59D0(v8, a3);
  result = *(_BYTE **)(v9 + 32);
  if ( *(_BYTE **)(v9 + 24) == result )
    return (_BYTE *)sub_CB6200(v9, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v9 + 32);
  return result;
}
