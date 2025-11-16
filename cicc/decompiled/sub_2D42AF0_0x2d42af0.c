// Function: sub_2D42AF0
// Address: 0x2d42af0
//
__int64 __fastcall sub_2D42AF0(
        __int64 (__fastcall *a1)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64),
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v16; // r8
  __int64 v17; // r9

  v16 = a6;
  v17 = (unsigned int)a7;
  LODWORD(a7) = (unsigned __int8)a8;
  return a1(a2, a3, a4, a5, v16, v17, a7, a9, a10, a11);
}
