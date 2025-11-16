// Function: sub_2C47CA0
// Address: 0x2c47ca0
//
__int64 __fastcall sub_2C47CA0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v5; // al
  __int64 *v6; // rdi
  _QWORD *v7; // rax
  __m128i *v8; // rdx
  _QWORD *v9; // rdi
  __m128i v10; // xmm0
  __int64 result; // rax
  _QWORD *v12; // rax
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  _QWORD *v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  _QWORD *v18; // rax
  __m128i *v19; // rdx
  __m128i v20; // xmm0

  v5 = *(_BYTE *)(a2 - 32);
  v6 = *a1;
  switch ( v5 )
  {
    case 18:
      return sub_2C47B10(v6, a2 - 40, *(_DWORD *)(a2 + 16) - 1, a4, a5);
    case 21:
    case 6:
      return sub_2C47B10(v6, a2 - 40, 2, a4, a5);
    case 19:
    case 13:
    case 35:
      return sub_2C47B10(v6, a2 - 40, 1, a4, a5);
    case 10:
      return sub_2C47B10(v6, a2 - 40, 0, a4, a5);
    case 4:
      if ( *(_BYTE *)(a2 + 120) == 13 )
      {
        if ( *(_DWORD *)(a2 + 80) == 1 )
        {
          result = 1;
          if ( *(_BYTE *)(**(_QWORD **)(a2 + 72) - 32LL) != 31 )
          {
            v12 = sub_CB72A0();
            v13 = (__m128i *)v12[4];
            if ( v12[3] - (_QWORD)v13 <= 0x52u )
            {
              sub_CB6200(
                (__int64)v12,
                "Result of VPInstruction::Add with EVL operand is not used by VPEVLBasedIVPHIRecipe\n",
                0x53u);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_43A14B0);
              v13[5].m128i_i8[2] = 10;
              v13[5].m128i_i16[0] = 25968;
              *v13 = si128;
              v13[1] = _mm_load_si128((const __m128i *)&xmmword_43A14C0);
              v13[2] = _mm_load_si128((const __m128i *)&xmmword_43A14D0);
              v13[3] = _mm_load_si128((const __m128i *)&xmmword_43A14E0);
              v13[4] = _mm_load_si128((const __m128i *)&xmmword_43A14F0);
              v12[4] += 83LL;
            }
            return 0;
          }
        }
        else
        {
          v18 = sub_CB72A0();
          v19 = (__m128i *)v18[4];
          if ( v18[3] - (_QWORD)v19 <= 0x34u )
          {
            sub_CB6200((__int64)v18, "EVL is used in VPInstruction:Add with multiple users\n", 0x35u);
          }
          else
          {
            v20 = _mm_load_si128((const __m128i *)&xmmword_43A1480);
            v19[3].m128i_i32[0] = 1936876915;
            v19[3].m128i_i8[4] = 10;
            *v19 = v20;
            v19[1] = _mm_load_si128((const __m128i *)&xmmword_43A1490);
            v19[2] = _mm_load_si128((const __m128i *)&xmmword_43A14A0);
            v18[4] += 53LL;
          }
          return 0;
        }
      }
      else
      {
        v15 = sub_CB72A0();
        v16 = (__m128i *)v15[4];
        if ( v15[3] - (_QWORD)v16 <= 0x33u )
        {
          sub_CB6200((__int64)v15, "EVL is used as an operand in non-VPInstruction::Add\n", 0x34u);
        }
        else
        {
          v17 = _mm_load_si128((const __m128i *)&xmmword_43A1450);
          v16[3].m128i_i32[0] = 174351425;
          *v16 = v17;
          v16[1] = _mm_load_si128((const __m128i *)&xmmword_43A1460);
          v16[2] = _mm_load_si128((const __m128i *)&xmmword_43A1470);
          v15[4] += 52LL;
        }
        return 0;
      }
      break;
    default:
      v7 = sub_CB72A0();
      v8 = (__m128i *)v7[4];
      v9 = v7;
      if ( v7[3] - (_QWORD)v8 <= 0x17u )
      {
        sub_CB6200((__int64)v7, "EVL has unexpected user\n", 0x18u);
        return 0;
      }
      else
      {
        v10 = _mm_load_si128((const __m128i *)&xmmword_43A1500);
        v8[1].m128i_i64[0] = 0xA72657375206465LL;
        result = 0;
        *v8 = v10;
        v9[4] += 24LL;
      }
      break;
  }
  return result;
}
