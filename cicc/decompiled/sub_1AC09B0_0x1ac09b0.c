// Function: sub_1AC09B0
// Address: 0x1ac09b0
//
__int64 __fastcall sub_1AC09B0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __int64 a7,
        unsigned __int8 a8,
        unsigned __int8 a9)
{
  char v10; // di
  __int64 result; // rax

  *(_QWORD *)a1 = a4;
  v10 = byte_4FB6320;
  *(_QWORD *)(a1 + 16) = a6;
  *(_BYTE *)(a1 + 32) = a8;
  if ( !a5 )
    a5 = v10;
  *(_BYTE *)(a1 + 8) = a5;
  *(_QWORD *)(a1 + 24) = a7;
  result = sub_1ABFDF0(a1 + 40, a2, a3, a4, a8, a9);
  *(_DWORD *)(a1 + 96) = -1;
  return result;
}
