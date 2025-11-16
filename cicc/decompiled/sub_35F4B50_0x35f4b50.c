// Function: sub_35F4B50
// Address: 0x35f4b50
//
unsigned __int64 __fastcall sub_35F4B50(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  int v4; // eax
  _WORD *v5; // rdx
  unsigned __int64 result; // rax
  __m128i v7; // xmm0
  __m128i si128; // xmm0

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  v5 = *(_WORD **)(a4 + 32);
  switch ( v4 )
  {
    case 0:
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v5;
      if ( result <= 5 )
      {
        result = sub_CB6200(a4, (unsigned __int8 *)".async", 6u);
      }
      else
      {
        *(_DWORD *)v5 = 2037604654;
        v5[2] = 25454;
        *(_QWORD *)(a4 + 32) += 6LL;
      }
      break;
    case 1:
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v5 <= 0xCu )
      {
        result = sub_CB6200(a4, ".async.global", 0xDu);
      }
      else
      {
        qmemcpy(v5, ".async.global", 13);
        *(_QWORD *)(a4 + 32) += 13LL;
        result = 0x672E636E7973612ELL;
      }
      break;
    case 2:
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v5;
      if ( result <= 0x11 )
      {
        result = sub_CB6200(a4, ".async.shared::cta", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44FE870);
        v5[8] = 24948;
        *(__m128i *)v5 = si128;
        *(_QWORD *)(a4 + 32) += 18LL;
      }
      break;
    case 3:
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v5;
      if ( result <= 0x15 )
      {
        result = sub_CB6200(a4, ".async.shared::cluster", 0x16u);
      }
      else
      {
        v7 = _mm_load_si128((const __m128i *)&xmmword_44FE870);
        *((_DWORD *)v5 + 4) = 1953723756;
        v5[10] = 29285;
        *(__m128i *)v5 = v7;
        *(_QWORD *)(a4 + 32) += 22LL;
      }
      break;
    case 4:
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v5 <= 5u )
      {
        result = sub_CB6200(a4, ".alias", 6u);
      }
      else
      {
        *(_DWORD *)v5 = 1768710446;
        v5[2] = 29537;
        *(_QWORD *)(a4 + 32) += 6LL;
        result = 29537;
      }
      break;
    default:
      BUG();
  }
  return result;
}
