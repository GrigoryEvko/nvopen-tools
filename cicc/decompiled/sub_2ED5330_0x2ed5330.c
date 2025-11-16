// Function: sub_2ED5330
// Address: 0x2ed5330
//
unsigned __int64 __fastcall sub_2ED5330(
        _BYTE *a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, char *, __int64),
        __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  void *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 result; // rax
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  size_t v13; // [rsp+8h] [rbp-18h]

  v6 = a3(a4, "MachineSinkingPass]", 18);
  v8 = *(void **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v8;
  if ( result < v7 )
  {
    result = sub_CB6200(a2, v9, v7);
  }
  else if ( v7 )
  {
    v13 = v7;
    result = (unsigned __int64)memcpy(v8, v9, v7);
    *(_QWORD *)(a2 + 32) += v13;
    if ( !*a1 )
      return result;
    goto LABEL_6;
  }
  if ( !*a1 )
    return result;
LABEL_6:
  v11 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 0x11u )
    return sub_CB6200(a2, "<enable-sink-fold>", 0x12u);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4451270);
  v11[1].m128i_i16[0] = 15972;
  *v11 = si128;
  *(_QWORD *)(a2 + 32) += 18LL;
  return 15972;
}
