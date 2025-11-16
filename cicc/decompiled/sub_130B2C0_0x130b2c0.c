// Function: sub_130B2C0
// Address: 0x130b2c0
//
__int64 __fastcall sub_130B2C0(
        int a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        _OWORD *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_13409B0(a2 + 68096, a5) )
    return 1;
  result = sub_130BDB0(a1, (int)a2 + 24, a5, a4, (int)a2 + 68096, a9, a10, a11, a12, (__int64)a7 + 8, a8);
  if ( (_BYTE)result )
    return 1;
  *(_BYTE *)(a2 + 17) = 0;
  *(_DWORD *)(a2 + 68240) = a6;
  *(_BYTE *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 68248) = a8;
  *(_QWORD *)(a2 + 68256) = a7;
  *a7 = 0;
  a7[1] = 0;
  a7[2] = 0;
  a7[3] = 0;
  a7[4] = 0;
  *(_QWORD *)(a2 + 68264) = a4;
  *(_QWORD *)(a2 + 68272) = a5;
  *(_QWORD *)a2 = a3;
  return result;
}
