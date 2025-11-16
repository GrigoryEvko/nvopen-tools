// Function: sub_1B75040
// Address: 0x1b75040
//
__int64 __fastcall sub_1B75040(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  result = sub_22077B0(360);
  if ( result )
  {
    *(_DWORD *)result = a3;
    *(_QWORD *)(result + 24) = result + 40;
    *(_QWORD *)(result + 72) = result + 88;
    *(_QWORD *)(result + 32) = 0x200000001LL;
    *(_QWORD *)(result + 184) = result + 200;
    *(_QWORD *)(result + 8) = a4;
    *(_DWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 40) = a2;
    *(_QWORD *)(result + 48) = a5;
    *(_QWORD *)(result + 80) = 0x400000000LL;
    *(_QWORD *)(result + 192) = 0x100000000LL;
    *(_QWORD *)(result + 216) = result + 232;
    *(_QWORD *)(result + 224) = 0x1000000000LL;
  }
  *a1 = result;
  return result;
}
