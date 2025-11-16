// Function: sub_CF63E0
// Address: 0xcf63e0
//
__int64 __fastcall sub_CF63E0(_QWORD *a1, unsigned __int8 *a2, const __m128i *a3, __int64 a4)
{
  int v4; // eax
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // r8
  unsigned int v8; // eax
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  __m128i v10; // [rsp+10h] [rbp-20h]
  __m128i v11; // [rsp+20h] [rbp-10h]

  v4 = *a2;
  if ( a3[3].m128i_i8[0] )
  {
    v5 = v4 - 34;
    v9 = _mm_loadu_si128(a3);
    v10 = _mm_loadu_si128(a3 + 1);
    v11 = _mm_loadu_si128(a3 + 2);
  }
  else
  {
    if ( (unsigned __int8)(v4 - 34) <= 0x33u )
    {
      v7 = 0x8000000000041LL;
      if ( _bittest64(&v7, (unsigned int)(v4 - 34)) )
      {
        v8 = sub_CF5230((__int64)a1, (__int64)a2, a4);
        return ((unsigned __int8)(v8 >> 6) | (unsigned __int8)((v8 >> 4) | v8 | (v8 >> 2))) & 3;
      }
    }
    v5 = v4 - 34;
    v9.m128i_i64[0] = 0;
    v9.m128i_i64[1] = -1;
    v10 = 0u;
    v11 = 0u;
    if ( v5 > 0x37 )
      return 0;
  }
  switch ( v5 )
  {
    case 0u:
    case 6u:
    case 0x33u:
      result = sub_CF52B0(a1, a2, (__int64)&v9, a4);
      break;
    case 4u:
      result = sub_CF62A0((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x1Bu:
      result = sub_CF6090((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x1Cu:
      result = sub_CF6120((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x1Eu:
      result = sub_CF61D0((__int64)a1, (__int64)a2, &v9);
      break;
    case 0x1Fu:
      result = sub_CF62C0((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x20u:
      result = sub_CF6350((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x2Fu:
      result = sub_CF6280((__int64)a1, (__int64)a2, &v9, a4);
      break;
    case 0x37u:
      result = sub_CF61F0((__int64)a1, (__int64)a2, &v9, a4);
      break;
    default:
      return 0;
  }
  return result;
}
