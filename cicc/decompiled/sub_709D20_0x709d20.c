// Function: sub_709D20
// Address: 0x709d20
//
__int64 __fastcall sub_709D20(const __m128i *a1, unsigned __int8 a2, _DWORD *a3)
{
  const __m128i *v4; // rbx
  __int64 v6; // rax
  unsigned int v7; // [rsp+Ch] [rbp-24h]

  v4 = a1;
  *a3 = 0;
  if ( sub_709CC0(a1, a2) || (unsigned int)sub_709C40(a1, a2) )
    return 0;
  if ( a2 <= 1u || a2 == 10 )
  {
    LOWORD(v7) = a1->m128i_i16[0];
    v6 = (v7 >> 26) & 0x1F;
  }
  else
  {
    if ( (a2 & 0xFD) != 9 && a2 != 2 )
    {
      if ( (unsigned __int8)(a2 - 3) <= 1u )
        goto LABEL_19;
      if ( (unsigned __int8)(a2 - 5) <= 1u )
      {
        if ( dword_4F07890 || a2 == 6 && unk_4F06930 == 106 )
          goto LABEL_19;
        if ( unk_4F06930 != 64 )
        {
          if ( unk_4F06930 == 113 )
          {
LABEL_27:
            if ( unk_4F07580 )
              v4 = (const __m128i *)((char *)&a1->m128i_u64[1] + 4);
            v6 = HIWORD(v4->m128i_i32[0]) & 0x7FFF;
            return v6 > 0;
          }
LABEL_15:
          *a3 = 1;
          return 1;
        }
      }
      else
      {
        if ( a2 != 14 && a2 > 8u )
        {
          if ( qword_4D040A0[a2] != 8 )
            goto LABEL_13;
LABEL_19:
          if ( unk_4F07580 )
            v4 = (const __m128i *)((char *)a1->m128i_i64 + 4);
          v6 = ((unsigned __int32)v4->m128i_i32[0] >> 20) & 0x7FF;
          return v6 > 0;
        }
        if ( a2 != 7 )
        {
LABEL_13:
          if ( (a2 == 8 || a2 == 13) && unk_4F06918 == 113 )
            goto LABEL_27;
          goto LABEL_15;
        }
        if ( unk_4F06924 != 64 )
          goto LABEL_15;
      }
      if ( unk_4F07580 )
        v4 = (const __m128i *)&a1->m128i_u64[1];
      v6 = v4->m128i_i32[0] & 0x7FFF;
      return v6 > 0;
    }
    v6 = (unsigned __int8)((unsigned __int32)a1->m128i_i32[0] >> 23);
  }
  return v6 > 0;
}
