// Function: sub_130B660
// Address: 0x130b660
//
__int64 __fastcall sub_130B660(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int8 a7,
        __int64 a8)
{
  unsigned __int64 v9; // rdx
  __int64 v11; // rsi
  __int64 v13; // rax
  unsigned int v14; // r14d

  v9 = *a3;
  if ( (v9 & 0x10000) != 0 )
    return 1;
  v11 = a2 + 24;
  v13 = a2 + 62264;
  if ( (v9 & 0x4000) == 0 )
    v13 = v11;
  v14 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64, _QWORD, __int64))(v13 + 16))(
          a1,
          v13,
          a3,
          a4,
          a5,
          a7,
          a8);
  if ( (_BYTE)v14 )
  {
    return 1;
  }
  else
  {
    _InterlockedAdd64((volatile signed __int64 *)(a2 + 8), (unsigned __int64)(a5 - a4) >> 12);
    *a3 = ((unsigned __int64)a6 << 20) | *a3 & 0xFFFFFFFFF00FFFFFLL;
    sub_1342130(a1, *(_QWORD *)(a2 + 68264), a3, a6, 0);
  }
  return v14;
}
