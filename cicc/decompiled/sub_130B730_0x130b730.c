// Function: sub_130B730
// Address: 0x130b730
//
__int64 __fastcall sub_130B730(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7)
{
  unsigned __int64 v8; // rdx
  __int64 v10; // rax
  unsigned int v11; // r14d

  v8 = *a3;
  if ( (v8 & 0x10000) != 0 )
    return 1;
  v10 = a2 + 62264;
  if ( (v8 & 0x4000) == 0 )
    v10 = a2 + 24;
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64, __int64))(v10 + 24))(
          a1,
          v10,
          a3,
          a4,
          a5,
          a7);
  if ( (_BYTE)v11 )
  {
    return 1;
  }
  else
  {
    _InterlockedSub64((volatile signed __int64 *)(a2 + 8), (unsigned __int64)(a4 - a5) >> 12);
    *a3 = ((unsigned __int64)a6 << 20) | *a3 & 0xFFFFFFFFF00FFFFFLL;
    sub_1342130(a1, *(_QWORD *)(a2 + 68264), a3, a6, 0);
  }
  return v11;
}
