// Function: sub_1C34B00
// Address: 0x1c34b00
//
__int64 __fastcall sub_1C34B00(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // rdi
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  __int64 v12; // rdi
  __int64 v13; // rdx

  sub_1C346E0(a1, *(_QWORD *)a2, a2);
  result = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 17;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 16) - 17) > 6u )
  {
    result = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
    {
      v4 = 0;
      v5 = 24LL * (unsigned int)result;
      do
      {
        while ( 1 )
        {
          v6 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v7 = *(_QWORD *)(v6 + v4);
          result = sub_1C346E0(a1, *(_QWORD *)v7, a2);
          if ( *(_BYTE *)(v7 + 16) == 4 )
            break;
          v4 += 24;
          if ( v5 == v4 )
            return result;
        }
        v8 = *(_BYTE *)(a2 + 16);
        if ( v8 <= 0x17u )
        {
          if ( v8 == 3 )
          {
            v9 = sub_1C31E60(a1, a2, 0);
          }
          else
          {
            v12 = *(_QWORD *)(a1 + 24);
            v13 = *(_QWORD *)(v12 + 24);
            if ( (unsigned __int64)(*(_QWORD *)(v12 + 16) - v13) <= 6 )
            {
              sub_16E7EE0(v12, "Error: ", 7u);
            }
            else
            {
              *(_DWORD *)v13 = 1869771333;
              *(_WORD *)(v13 + 4) = 14962;
              *(_BYTE *)(v13 + 6) = 32;
              *(_QWORD *)(v12 + 24) += 7LL;
            }
            v9 = *(_QWORD *)(a1 + 24);
          }
        }
        else
        {
          v9 = sub_1C321C0(a1, a2, 0);
        }
        v10 = *(__m128i **)(v9 + 24);
        if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0x1Du )
        {
          sub_16E7EE0(v9, "blockaddress is not supported\n", 0x1Eu);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42D0970);
          qmemcpy(&v10[1], "not supported\n", 14);
          *v10 = si128;
          *(_QWORD *)(v9 + 24) += 30LL;
        }
        result = sub_1C31880(a1);
        v4 += 24;
      }
      while ( v5 != v4 );
    }
  }
  return result;
}
