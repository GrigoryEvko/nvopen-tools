// Function: sub_709F60
// Address: 0x709f60
//
_BOOL8 __fastcall sub_709F60(__m128i *a1, unsigned __int8 a2, int a3, int a4)
{
  __m128i *v7; // rax
  __int32 v8; // esi
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v10[0] = 0;
  *a1 = 0;
  if ( a3 )
  {
    a1->m128i_i32[0] = 2139095040;
    if ( a2 == 2 )
      goto LABEL_3;
  }
  else
  {
    a1->m128i_i32[0] = 2143289344;
    if ( a2 == 2 )
      goto LABEL_17;
  }
  sub_709EF0(a1, 2u, a1, a2, v10, (_DWORD *)v10 + 1);
LABEL_3:
  if ( a4 )
    goto LABEL_6;
  if ( a3 )
  {
    a4 = 1;
    goto LABEL_6;
  }
LABEL_17:
  if ( !a4 )
    return v10[0] == 0;
LABEL_6:
  v7 = a1;
  if ( unk_4F07580 )
  {
LABEL_10:
    if ( a2 == 10 || (v8 = v7->m128i_i32[0], a2 <= 1u) )
    {
      v7->m128i_i32[0] |= a4 & 0x7FF;
      return v10[0] == 0;
    }
    goto LABEL_12;
  }
  if ( (unsigned __int8)(a2 - 9) <= 1u || a2 <= 1u )
  {
    v7 = (__m128i *)((char *)a1 - 4);
    goto LABEL_10;
  }
  if ( a2 == 2 )
  {
    v7 = (__m128i *)((char *)a1 + (unk_4F06A48 & 0xFFFFFFFFFFFFFFFCLL) - 4);
    v8 = v7->m128i_i32[0];
  }
  else if ( (unsigned __int8)(a2 - 3) <= 1u )
  {
    v7 = (__m128i *)((char *)a1 + (unk_4F06A38 & 0xFFFFFFFFFFFFFFFCLL) - 4);
    v8 = v7->m128i_i32[0];
  }
  else if ( (unsigned __int8)(a2 - 5) <= 1u )
  {
    v7 = (__m128i *)((char *)a1 + (unk_4F06A28 & 0xFFFFFFFFFFFFFFFCLL) - 4);
    v8 = v7->m128i_i32[0];
  }
  else if ( a2 == 7 )
  {
    v7 = (__m128i *)((char *)a1 + (unk_4F06A18 & 0xFFFFFFFFFFFFFFFCLL) - 4);
    v8 = v7->m128i_i32[0];
  }
  else if ( a2 == 8 )
  {
    v7 = (__m128i *)((char *)a1 + (unk_4F06A08 & 0xFFFFFFFFFFFFFFFCLL) - 4);
    v8 = v7->m128i_i32[0];
  }
  else
  {
    if ( a2 != 11 )
    {
      v7 = (__m128i *)((char *)a1->m128i_i64 + 4);
      if ( a2 != 12 )
      {
        if ( a2 != 13 )
          sub_721090(a1);
        v7 = (__m128i *)((char *)&a1->m128i_u64[1] + 4);
      }
    }
    v8 = v7->m128i_i32[0];
  }
LABEL_12:
  if ( (a2 & 0xFD) == 9 || a2 == 2 )
    a4 = v8 | a4 & 0x7FFFFF;
  v7->m128i_i32[0] = a4;
  return v10[0] == 0;
}
