// Function: sub_877070
// Address: 0x877070
//
__int64 sub_877070()
{
  __int64 result; // rax

  result = sub_823970(80);
  *(_WORD *)(result + 72) &= 0x8000u;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_DWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_BYTE *)(result + 74) = 0;
  *(_WORD *)(result + 76) = 0;
  return result;
}
