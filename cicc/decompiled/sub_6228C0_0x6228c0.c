// Function: sub_6228C0
// Address: 0x6228c0
//
char *__fastcall sub_6228C0(const __m128i *a1)
{
  __int64 i; // rax
  int v2; // eax

  for ( i = a1[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = sub_620E90((__int64)a1);
  return sub_622500(a1 + 11, v2);
}
