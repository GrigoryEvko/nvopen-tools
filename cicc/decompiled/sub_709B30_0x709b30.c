// Function: sub_709B30
// Address: 0x709b30
//
__int64 __fastcall sub_709B30(__int64 a1, const __m128i *a2)
{
  __int128 dest; // [rsp+0h] [rbp-10h] BYREF

  dest = 0;
  if ( (unsigned __int8)a1 <= 1u || (_BYTE)a1 == 10 )
  {
    sub_12F9660(a2->m128i_u16[0], &dest);
    return dest;
  }
  else
  {
    if ( (a1 & 0xFD) == 9 || (_BYTE)a1 == 2 )
    {
      sub_12F9710(a2->m128i_u32[0], &dest);
      return dest;
    }
    if ( (unsigned __int8)(a1 - 3) > 1u )
    {
      if ( (unsigned __int8)(a1 - 5) <= 1u )
      {
        if ( !dword_4F07890 )
        {
          if ( unk_4F06930 == 64 )
          {
            sub_12F98E0(a2, &dest);
          }
          else
          {
            if ( unk_4F06930 != 113 )
              sub_721090(a1);
            memcpy(&dest, a2, unk_4F06A28);
          }
          return dest;
        }
      }
      else if ( (_BYTE)a1 == 14 || (unsigned __int8)a1 <= 8u || qword_4D040A0[(unsigned __int8)a1] != 8 )
      {
        return _mm_loadu_si128(a2).m128i_i64[0];
      }
    }
    sub_12F97D0(a2->m128i_i64[0], &dest);
    return dest;
  }
}
