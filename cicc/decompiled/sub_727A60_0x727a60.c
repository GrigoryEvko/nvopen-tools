// Function: sub_727A60
// Address: 0x727a60
//
__int64 sub_727A60()
{
  __int64 result; // rax

  result = sub_7279A0(80);
  *(_WORD *)(result + 72) &= 0xFE00u;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0xFFFFFFFF00000000LL;
  *(_DWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  return result;
}
