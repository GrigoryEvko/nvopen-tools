// Function: sub_67B9F0
// Address: 0x67b9f0
//
__int64 sub_67B9F0()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4D039F8;
  if ( !qword_4D039F8 || dword_4D03A00 == -1 )
    result = sub_823020((unsigned int)dword_4D03A00, 200);
  else
    qword_4D039F8 = *(_QWORD *)(qword_4D039F8 + 8);
  *(_DWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  v1 = unk_4F077C8;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = 0;
  *(_QWORD *)(result + 80) = 0;
  *(_QWORD *)(result + 88) = 0;
  *(_QWORD *)(result + 96) = v1;
  *(_DWORD *)(result + 104) = 0;
  *(_QWORD *)(result + 112) = 0;
  *(_QWORD *)(result + 120) = 0;
  *(_QWORD *)(result + 128) = 0;
  *(_QWORD *)(result + 136) = v1;
  *(_DWORD *)(result + 144) = 0;
  *(_QWORD *)(result + 152) = 0;
  *(_QWORD *)(result + 160) = 0;
  *(_QWORD *)(result + 168) = 0;
  *(_DWORD *)(result + 176) = 0;
  *(_BYTE *)(result + 180) = 3;
  *(_QWORD *)(result + 184) = 0;
  *(_QWORD *)(result + 192) = 0;
  return result;
}
