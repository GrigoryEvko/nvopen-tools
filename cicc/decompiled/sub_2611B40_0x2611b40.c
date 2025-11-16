// Function: sub_2611B40
// Address: 0x2611b40
//
unsigned __int64 __fastcall sub_2611B40(
        __int64 a1,
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
  size_t v12; // [rsp+8h] [rbp-18h]

  v6 = a3(a4, "InlinerPass]", 11);
  v8 = *(void **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v8;
  if ( result < v7 )
  {
    result = sub_CB6200(a2, v9, v7);
  }
  else if ( v7 )
  {
    v12 = v7;
    result = (unsigned __int64)memcpy(v8, v9, v7);
    *(_QWORD *)(a2 + 32) += v12;
    if ( !*(_BYTE *)(a1 + 8) )
      return result;
    goto LABEL_6;
  }
  if ( !*(_BYTE *)(a1 + 8) )
    return result;
LABEL_6:
  v11 = *(__m128i **)(a2 + 32);
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v11;
  if ( result <= 0xF )
    return sub_CB6200(a2, "<only-mandatory>", 0x10u);
  *v11 = _mm_load_si128((const __m128i *)&xmmword_438C660);
  *(_QWORD *)(a2 + 32) += 16LL;
  return result;
}
