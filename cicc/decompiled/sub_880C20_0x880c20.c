// Function: sub_880C20
// Address: 0x880c20
//
__int64 sub_880C20()
{
  __int64 result; // rax

  result = sub_823970(32);
  *(_BYTE *)(result + 28) &= 0xF0u;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_DWORD *)(result + 24) = 0;
  return result;
}
