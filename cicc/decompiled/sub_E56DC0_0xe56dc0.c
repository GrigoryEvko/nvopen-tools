// Function: sub_E56DC0
// Address: 0xe56dc0
//
_BYTE *__fastcall sub_E56DC0(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __m128i *v7; // rdx
  __m128i si128; // xmm0

  sub_E997E0();
  result = *(_BYTE **)(a1 + 104);
  if ( result )
  {
    v3 = *((_QWORD *)result + 4);
    v4 = *(_QWORD *)v3;
    if ( !*(_QWORD *)v3 )
    {
      if ( (*(_BYTE *)(v3 + 9) & 0x70) != 0x20 || *(char *)(v3 + 8) < 0 )
        BUG();
      *(_BYTE *)(v3 + 8) |= 8u;
      v4 = sub_E807D0(*(_QWORD *)(v3 + 24));
      *(_QWORD *)v3 = v4;
    }
    v5 = sub_E99A60(a1, *(_QWORD *)(v4 + 8));
    sub_E98210(a1, v5);
    v6 = *(_QWORD *)(a1 + 304);
    v7 = *(__m128i **)(v6 + 32);
    if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0x10u )
    {
      sub_CB6200(v6, "\t.seh_handlerdata", 0x11u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F8C0);
      v7[1].m128i_i8[0] = 97;
      *v7 = si128;
      *(_QWORD *)(v6 + 32) += 17LL;
    }
    return sub_E4D880(a1);
  }
  return result;
}
