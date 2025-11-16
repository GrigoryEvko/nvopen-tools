// Function: sub_1C32630
// Address: 0x1c32630
//
__int64 __fastcall sub_1C32630(__int64 a1, char *a2, __int64 a3)
{
  __int64 v4; // rax
  __m128i *v5; // rdx
  __int64 v6; // r12
  __m128i si128; // xmm0
  size_t v8; // rax
  _BYTE *v9; // rdi
  size_t v10; // r15
  _BYTE *v11; // rax

  v4 = sub_1C321C0(a1, a3, 0);
  v5 = *(__m128i **)(v4 + 24);
  v6 = v4;
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0x14u )
  {
    v6 = sub_16E7EE0(v4, "Illegal instruction: ", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42D0530);
    v5[1].m128i_i32[0] = 980316009;
    v5[1].m128i_i8[4] = 32;
    *v5 = si128;
    *(_QWORD *)(v4 + 24) += 21LL;
  }
  v8 = strlen(a2);
  v9 = *(_BYTE **)(v6 + 24);
  v10 = v8;
  v11 = *(_BYTE **)(v6 + 16);
  if ( v10 <= v11 - v9 )
  {
    if ( v10 )
    {
      memcpy(v9, a2, v10);
      v11 = *(_BYTE **)(v6 + 16);
      v9 = (_BYTE *)(v10 + *(_QWORD *)(v6 + 24));
      *(_QWORD *)(v6 + 24) = v9;
    }
    if ( v11 != v9 )
      goto LABEL_7;
LABEL_10:
    sub_16E7EE0(v6, "\n", 1u);
    return sub_1C31880(a1);
  }
  v6 = sub_16E7EE0(v6, a2, v10);
  v9 = *(_BYTE **)(v6 + 24);
  if ( *(_BYTE **)(v6 + 16) == v9 )
    goto LABEL_10;
LABEL_7:
  *v9 = 10;
  ++*(_QWORD *)(v6 + 24);
  return sub_1C31880(a1);
}
