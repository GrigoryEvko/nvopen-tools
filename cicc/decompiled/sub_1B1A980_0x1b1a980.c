// Function: sub_1B1A980
// Address: 0x1b1a980
//
_QWORD *__fastcall sub_1B1A980(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  _QWORD *result; // rax

  switch ( *(_DWORD *)(a3 + 32) )
  {
    case 0:
    case 9:
      result = sub_1B1A460(
                 a1,
                 a2,
                 0x34u,
                 a4,
                 (*(_DWORD *)(a3 + 36) == 6) | ((unsigned __int64)a5 << 16),
                 a6,
                 a7,
                 a8,
                 (__int64)a4,
                 0,
                 0);
      break;
    case 1:
      result = sub_1B1A460(a1, a2, 0xBu, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 2:
      result = sub_1B1A460(a1, a2, 0xFu, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 3:
      result = sub_1B1A460(a1, a2, 0x1Bu, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 4:
      result = sub_1B1A460(a1, a2, 0x1Au, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 5:
      result = sub_1B1A460(a1, a2, 0x1Cu, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 6:
      result = sub_1B1A460(
                 a1,
                 a2,
                 0x33u,
                 a4,
                 (((*(_DWORD *)(a3 + 36) - 2) & 0xFFFFFFFD) == 0)
               | ((unsigned __int64)((unsigned int)(*(_DWORD *)(a3 + 36) - 3) <= 1) << 8)
               | ((unsigned __int64)a5 << 16),
                 a6,
                 a7,
                 a8,
                 (__int64)a4,
                 0,
                 0);
      break;
    case 7:
      result = sub_1B1A460(a1, a2, 0xCu, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
    case 8:
      result = sub_1B1A460(a1, a2, 0x10u, a4, (unsigned __int64)a5 << 16, a6, a7, a8, (__int64)a4, 0, 0);
      break;
  }
  return result;
}
