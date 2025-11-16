// Function: sub_27CAA50
// Address: 0x27caa50
//
_BYTE *__fastcall sub_27CAA50(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rdx
  _QWORD *v6; // rdx
  __int64 v7; // rdx
  void *v8; // rdx
  void *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  _BYTE *result; // rax

  v2 = a2;
  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x14u )
  {
    sub_CB6200(a2, "InductiveRangeCheck:\n", 0x15u);
    v5 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) > 8 )
      goto LABEL_3;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42BEB70);
    v3[1].m128i_i32[0] = 980116325;
    v3[1].m128i_i8[4] = 10;
    *v3 = si128;
    v5 = *(_QWORD *)(a2 + 32) + 21LL;
    *(_QWORD *)(a2 + 32) = v5;
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) > 8 )
    {
LABEL_3:
      *(_BYTE *)(v5 + 8) = 32;
      *(_QWORD *)v5 = 0x3A6E696765422020LL;
      *(_QWORD *)(a2 + 32) += 9LL;
      goto LABEL_4;
    }
  }
  sub_CB6200(a2, "  Begin: ", 9u);
LABEL_4:
  sub_D955C0(*a1, a2);
  v6 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 7u )
  {
    sub_CB6200(a2, "  Step: ", 8u);
  }
  else
  {
    *v6 = 0x203A706574532020LL;
    *(_QWORD *)(a2 + 32) += 8LL;
  }
  sub_D955C0(a1[1], a2);
  v7 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v7) <= 6 )
  {
    sub_CB6200(a2, "  End: ", 7u);
  }
  else
  {
    *(_DWORD *)v7 = 1850023968;
    *(_WORD *)(v7 + 4) = 14948;
    *(_BYTE *)(v7 + 6) = 32;
    *(_QWORD *)(a2 + 32) += 7LL;
  }
  sub_D955C0(a1[2], a2);
  v8 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 0xCu )
  {
    sub_CB6200(a2, "\n  CheckUse: ", 0xDu);
  }
  else
  {
    qmemcpy(v8, "\n  CheckUse: ", 13);
    *(_QWORD *)(a2 + 32) += 13LL;
  }
  sub_A69870(*(_QWORD *)(a1[3] + 24), (_BYTE *)a2, 0);
  v9 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 <= 9u )
  {
    v2 = sub_CB6200(a2, " Operand: ", 0xAu);
  }
  else
  {
    qmemcpy(v9, " Operand: ", 10);
    *(_QWORD *)(a2 + 32) += 10LL;
  }
  v10 = sub_BD2910(a1[3]);
  v11 = sub_CB59D0(v2, v10);
  result = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == result )
    return (_BYTE *)sub_CB6200(v11, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(v11 + 32);
  return result;
}
