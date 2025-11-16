// Function: sub_880C60
// Address: 0x880c60
//
__int64 sub_880C60()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = sub_823970(128);
  *(_QWORD *)result = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  *(_WORD *)(result + 80) &= 0xF000u;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = 0;
  *(_QWORD *)(result + 84) = v1;
  *(_QWORD *)(result + 92) = v1;
  *(_QWORD *)(result + 104) = 0;
  *(_QWORD *)(result + 112) = 0;
  *(_QWORD *)(result + 120) = 0;
  return result;
}
