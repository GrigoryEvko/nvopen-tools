// Function: sub_18F2CB0
// Address: 0x18f2cb0
//
__m128i *__fastcall sub_18F2CB0(__m128i *a1, __int64 a2)
{
  __int64 v3; // rax

  if ( *(_BYTE *)(a2 + 16) == 78
    && (v3 = *(_QWORD *)(a2 - 24), !*(_BYTE *)(v3 + 16))
    && (*(_BYTE *)(v3 + 33) & 0x20) != 0
    && (unsigned int)(*(_DWORD *)(v3 + 36) - 133) <= 3 )
  {
    sub_141F670(a1, a2);
    return a1;
  }
  else
  {
    a1->m128i_i64[0] = 0;
    a1->m128i_i64[1] = -1;
    a1[1].m128i_i64[0] = 0;
    a1[1].m128i_i64[1] = 0;
    a1[2].m128i_i64[0] = 0;
    return a1;
  }
}
