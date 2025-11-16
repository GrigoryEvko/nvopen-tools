// Function: sub_AF2C10
// Address: 0xaf2c10
//
__int64 __fastcall sub_AF2C10(
        __int64 a1,
        int a2,
        int a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v10; // eax
  __int64 result; // rax

  sub_B971C0(a1, a2, 11, a3, a7, a8, 0, 0);
  *(_WORD *)(a1 + 2) = 40;
  v10 = *(_DWORD *)(a4 + 8);
  *(_DWORD *)(a1 + 24) = v10;
  if ( v10 > 0x40 )
  {
    result = sub_C43780(a1 + 16, a4);
  }
  else
  {
    result = *(_QWORD *)a4;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)a4;
  }
  *(_DWORD *)(a1 + 4) = a5;
  return result;
}
