// Function: sub_D95B00
// Address: 0xd95b00
//
_BYTE *__fastcall sub_D95B00(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // r12
  __m128i v6; // xmm0
  _BYTE *v7; // rax
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // r12
  _BYTE *result; // rax
  __int64 v12; // rax
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  _DWORD *v15; // rdx

  if ( *(_DWORD *)(a1 + 36) == 32 )
  {
    v12 = sub_CB69B0(a2, a3);
    v13 = *(__m128i **)(v12 + 32);
    v10 = v12;
    if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0x10u )
    {
      v10 = sub_CB6200(v12, "Equal predicate: ", 0x11u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F74EA0);
      v13[1].m128i_i8[0] = 32;
      *v13 = si128;
      *(_QWORD *)(v12 + 32) += 17LL;
    }
    sub_D955C0(*(_QWORD *)(a1 + 40), v10);
    v15 = *(_DWORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v15 <= 3u )
    {
      v10 = sub_CB6200(v10, " == ", 4u);
    }
    else
    {
      *v15 = 540884256;
      *(_QWORD *)(v10 + 32) += 4LL;
    }
  }
  else
  {
    v3 = sub_CB69B0(a2, a3);
    v4 = *(__m128i **)(v3 + 32);
    v5 = v3;
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x12u )
    {
      v5 = sub_CB6200(v3, "Compare predicate: ", 0x13u);
    }
    else
    {
      v6 = _mm_load_si128((const __m128i *)&xmmword_3F74EB0);
      v4[1].m128i_i8[2] = 32;
      v4[1].m128i_i16[0] = 14949;
      *v4 = v6;
      *(_QWORD *)(v3 + 32) += 19LL;
    }
    sub_D955C0(*(_QWORD *)(a1 + 40), v5);
    v7 = *(_BYTE **)(v5 + 32);
    if ( *(_BYTE **)(v5 + 24) == v7 )
    {
      v5 = sub_CB6200(v5, (unsigned __int8 *)" ", 1u);
    }
    else
    {
      *v7 = 32;
      ++*(_QWORD *)(v5 + 32);
    }
    v8 = sub_B52E10(v5, *(_DWORD *)(a1 + 36));
    v9 = *(_WORD **)(v8 + 32);
    v10 = v8;
    if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
    {
      v10 = sub_CB6200(v8, (unsigned __int8 *)") ", 2u);
    }
    else
    {
      *v9 = 8233;
      *(_QWORD *)(v8 + 32) += 2LL;
    }
  }
  sub_D955C0(*(_QWORD *)(a1 + 48), v10);
  result = *(_BYTE **)(v10 + 32);
  if ( *(_BYTE **)(v10 + 24) == result )
    return (_BYTE *)sub_CB6200(v10, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v10 + 32);
  return result;
}
