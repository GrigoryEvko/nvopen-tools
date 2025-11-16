// Function: sub_727AD0
// Address: 0x727ad0
//
__int64 sub_727AD0()
{
  __int64 result; // rax

  result = sub_7279A0(32);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0xFFFFFFFF00000000LL;
  *(_DWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  return result;
}
