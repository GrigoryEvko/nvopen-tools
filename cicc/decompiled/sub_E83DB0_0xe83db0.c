// Function: sub_E83DB0
// Address: 0xe83db0
//
__int64 __fastcall sub_E83DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  __int64 v7; // r8
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __m128i *v11; // rsi
  __int64 v12; // rax
  bool v13; // zf
  void *v14; // rax
  void *v15; // rax
  __m128i v16; // [rsp+0h] [rbp-20h] BYREF

  v6 = a3;
  v7 = *(_QWORD *)(a1 + 296);
  if ( (_DWORD)a3 == 14 )
  {
    v9 = *(_QWORD *)(a1 + 288);
    v10 = *(_QWORD *)(v7 + 24);
    v16.m128i_i64[0] = a2;
    v11 = *(__m128i **)(v10 + 152);
    v16.m128i_i64[1] = *(_QWORD *)(v9 + 8);
    if ( v11 == *(__m128i **)(v10 + 160) )
    {
      sub_E83C20(v10 + 144, v11, &v16);
    }
    else
    {
      if ( v11 )
      {
        *v11 = _mm_loadu_si128(&v16);
        v11 = *(__m128i **)(v10 + 152);
      }
      *(_QWORD *)(v10 + 152) = v11 + 1;
    }
    return 1;
  }
  else
  {
    sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, v7, a6);
    switch ( v6 )
    {
      case 0:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 17:
      case 22:
      case 24:
      case 28:
      case 29:
        return 0;
      case 1:
        *(_WORD *)(a2 + 12) |= 0x400u;
        return 1;
      case 9:
        *(_BYTE *)(a2 + 8) |= 0x20u;
        result = 1;
        *(_WORD *)(a2 + 12) &= ~1u;
        return result;
      case 16:
        LOWORD(v12) = *(_WORD *)(a2 + 12) | 0x20;
        v13 = *(_QWORD *)a2 == 0;
        *(_WORD *)(a2 + 12) = v12;
        if ( !v13 )
          return 1;
        if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
          goto LABEL_20;
        *(_BYTE *)(a2 + 8) |= 8u;
        v14 = sub_E807D0(*(_QWORD *)(a2 + 24));
        *(_QWORD *)a2 = v14;
        if ( v14 )
          return 1;
        v12 = *(unsigned __int16 *)(a2 + 12);
LABEL_20:
        *(_WORD *)(a2 + 12) = v12 | 1;
        result = 1;
        break;
      case 18:
      case 23:
        *(_WORD *)(a2 + 12) |= 0x20u;
        return 1;
      case 19:
        *(_WORD *)(a2 + 12) |= 0x100u;
        return 1;
      case 20:
        *(_WORD *)(a2 + 12) |= 0x200u;
        return 1;
      case 21:
        *(_BYTE *)(a2 + 8) |= 0x60u;
        return 1;
      case 25:
        *(_WORD *)(a2 + 12) |= 0x80u;
        return 1;
      case 26:
        if ( *(_QWORD *)a2 )
          return 1;
        if ( (*(_BYTE *)(a2 + 9) & 0x70) == 0x20 && *(char *)(a2 + 8) >= 0 )
        {
          *(_BYTE *)(a2 + 8) |= 8u;
          v15 = sub_E807D0(*(_QWORD *)(a2 + 24));
          *(_QWORD *)a2 = v15;
          if ( v15 )
            return 1;
        }
        *(_WORD *)(a2 + 12) |= 0x40u;
        return 1;
      case 27:
        *(_WORD *)(a2 + 12) |= 0xC0u;
        return 1;
      default:
        return 1;
    }
  }
  return result;
}
