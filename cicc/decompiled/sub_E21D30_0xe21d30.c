// Function: sub_E21D30
// Address: 0xe21d30
//
__int64 __fastcall sub_E21D30(__int64 a1, __int64 *a2)
{
  unsigned __int8 *v2; // rdx
  __int64 v3; // rcx
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rax
  unsigned __int8 v8; // al
  int v9; // eax
  int v10; // edi
  __m128i si128; // xmm0
  __m128i v12; // xmm0
  _BYTE v13[8]; // [rsp-8h] [rbp-8h] BYREF

  v2 = (unsigned __int8 *)a2[1];
  v3 = *a2;
  v5 = *a2 - 1;
  result = *v2;
  a2[1] = (__int64)(v2 + 1);
  *a2 = v5;
  if ( v3 && (_BYTE)result == 63 )
  {
    if ( v5 )
    {
      v7 = (char)v2[1];
      if ( (_BYTE)v7 != 36 )
      {
        if ( (unsigned int)((char)v7 - 48) <= 9 )
        {
          a2[1] = (__int64)(v2 + 2);
          *a2 = v3 - 2;
          return *((unsigned __int8 *)&jpt_E20DF5[29] + v7 + 3);
        }
        else if ( (unsigned __int8)(v7 - 97) <= 0x19u )
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F7C850);
          *(_WORD *)&v13[-8] = -1287;
          *(_QWORD *)&v13[-16] = 0xF8F7F6F5F4F3F2F1LL;
          a2[1] = (__int64)(v2 + 2);
          *a2 = v3 - 2;
          *(__m128i *)&v13[-32] = si128;
          return (unsigned __int8)v13[(char)(v7 - 97) - 32];
        }
        else
        {
          v8 = v7 - 65;
          if ( v8 <= 0x19u )
          {
            *(_QWORD *)&v13[-16] = 0xD8D7D6D5D4D3D2D1LL;
            v12 = _mm_load_si128((const __m128i *)&xmmword_3F7C860);
            *(_WORD *)&v13[-8] = -9511;
            a2[1] = (__int64)(v2 + 2);
            *a2 = v3 - 2;
            *(__m128i *)&v13[-32] = v12;
            return (unsigned __int8)v13[(char)v8 - 32];
          }
          else
          {
            *(_BYTE *)(a1 + 8) = 1;
            return 0;
          }
        }
      }
      a2[1] = (__int64)(v2 + 2);
      *a2 = v3 - 2;
      if ( (unsigned __int64)(v3 - 2) > 1 )
      {
        v9 = v2[2] - 65;
        if ( (unsigned __int8)(v2[2] - 65) <= 0xFu )
        {
          v10 = v2[3] - 65;
          if ( (unsigned __int8)(v2[3] - 65) <= 0xFu )
          {
            a2[1] = (__int64)(v2 + 4);
            *a2 = v3 - 4;
            return v10 | (unsigned int)(16 * v9);
          }
        }
      }
    }
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return result;
}
