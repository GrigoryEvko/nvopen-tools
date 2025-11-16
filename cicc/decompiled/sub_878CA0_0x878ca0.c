// Function: sub_878CA0
// Address: 0x878ca0
//
__int64 sub_878CA0()
{
  __int64 result; // rax

  result = qword_4F5FFD8;
  if ( qword_4F5FFD8 )
    qword_4F5FFD8 = *(_QWORD *)(qword_4F5FFD8 + 24);
  else
    result = sub_823970(104);
  *(_QWORD *)result = 0;
  *(_DWORD *)(result + 8) = -1;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_BYTE *)(result + 40) = 0;
  *(_WORD *)(result + 42) = 0;
  *(_DWORD *)(result + 44) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = 0;
  *(_QWORD *)(result + 80) = 0;
  *(_QWORD *)(result + 88) = 0;
  *(_QWORD *)(result + 96) = 0;
  return result;
}
