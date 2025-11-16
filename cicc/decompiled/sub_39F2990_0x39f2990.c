// Function: sub_39F2990
// Address: 0x39f2990
//
__int64 __fastcall sub_39F2990(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r8
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __m128i *v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __m128i v12; // [rsp+0h] [rbp-20h] BYREF

  v4 = *(_QWORD *)(a1 + 264);
  if ( a3 == 10 )
  {
    v6 = *(unsigned int *)(a1 + 120);
    v12.m128i_i64[0] = a2;
    v7 = 0;
    if ( (_DWORD)v6 )
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v6 - 32);
    v12.m128i_i64[1] = v7;
    v8 = *(__m128i **)(v4 + 88);
    if ( v8 == *(__m128i **)(v4 + 96) )
    {
      sub_39F2800(v4 + 80, v8, &v12);
    }
    else
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(&v12);
        v8 = *(__m128i **)(v4 + 88);
      }
      *(_QWORD *)(v4 + 88) = v8 + 1;
    }
    return 1;
  }
  else
  {
    sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
    switch ( a3 )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 9:
      case 10:
      case 11:
      case 13:
      case 18:
      case 20:
        return 0;
      case 8:
        *(_BYTE *)(a2 + 8) |= 0x10u;
        result = 1;
        *(_WORD *)(a2 + 12) &= ~1u;
        return result;
      case 12:
        LOWORD(v9) = *(_WORD *)(a2 + 12) | 0x20;
        *(_WORD *)(a2 + 12) = v9;
        if ( (*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          return 1;
        if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8 )
          goto LABEL_24;
        *(_BYTE *)(a2 + 8) |= 4u;
        v10 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
        *(_QWORD *)a2 = v10 | *(_QWORD *)a2 & 7LL;
        if ( v10 )
          return 1;
        v9 = *(unsigned __int16 *)(a2 + 12);
LABEL_24:
        *(_WORD *)(a2 + 12) = v9 | 1;
        result = 1;
        break;
      case 14:
      case 19:
        *(_WORD *)(a2 + 12) |= 0x20u;
        return 1;
      case 15:
        *(_WORD *)(a2 + 12) |= 0x100u;
        return 1;
      case 16:
        *(_WORD *)(a2 + 12) |= 0x200u;
        return 1;
      case 17:
        *(_BYTE *)(a2 + 8) |= 0x30u;
        return 1;
      case 21:
        *(_WORD *)(a2 + 12) |= 0x80u;
        return 1;
      case 22:
        if ( (*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          return 1;
        if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(a2 + 8) |= 4u;
          v11 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
          *(_QWORD *)a2 = v11 | *(_QWORD *)a2 & 7LL;
          if ( v11 )
            return 1;
        }
        *(_WORD *)(a2 + 12) |= 0x40u;
        return 1;
      case 23:
        *(_WORD *)(a2 + 12) |= 0xC0u;
        return 1;
      default:
        return 1;
    }
  }
  return result;
}
