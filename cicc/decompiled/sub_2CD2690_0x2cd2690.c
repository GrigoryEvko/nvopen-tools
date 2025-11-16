// Function: sub_2CD2690
// Address: 0x2cd2690
//
__int64 __fastcall sub_2CD2690(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r13
  __int64 result; // rax
  _BYTE *v13; // rax
  __m128i si128; // xmm0

  v6 = a3(a4, "LowerOpsPass]", 12);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(a2 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v10 = *(_BYTE **)(a2 + 24);
    v8 = *(_BYTE **)(a2 + 32);
LABEL_3:
    if ( v8 != v10 )
      goto LABEL_4;
    goto LABEL_8;
  }
  if ( !v7 )
    goto LABEL_3;
  memcpy(v8, v9, v7);
  v13 = *(_BYTE **)(a2 + 24);
  v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
  *(_QWORD *)(a2 + 32) = v8;
  if ( v8 != v13 )
  {
LABEL_4:
    *v8 = 60;
    result = *(_QWORD *)(a2 + 32) + 1LL;
    *(_QWORD *)(a2 + 32) = result;
    if ( !*a1 )
      goto LABEL_5;
LABEL_9:
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 0x12 )
    {
      sub_CB6200(a2, "enable-optimization", 0x13u);
      result = *(_QWORD *)(a2 + 32);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_444AA70);
      *(_BYTE *)(result + 18) = 110;
      *(_WORD *)(result + 16) = 28521;
      *(__m128i *)result = si128;
      result = *(_QWORD *)(a2 + 32) + 19LL;
      *(_QWORD *)(a2 + 32) = result;
    }
    goto LABEL_5;
  }
LABEL_8:
  sub_CB6200(a2, "<", 1u);
  result = *(_QWORD *)(a2 + 32);
  if ( *a1 )
    goto LABEL_9;
LABEL_5:
  if ( *(_QWORD *)(a2 + 24) == result )
    return sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *(_BYTE *)result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
