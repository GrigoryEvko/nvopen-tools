// Function: sub_7F9300
// Address: 0x7f9300
//
__int64 sub_7F9300()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = (__int64)qword_4D03F70;
  if ( qword_4D03F70 )
    qword_4D03F70 = (_QWORD *)*qword_4D03F70;
  else
    result = sub_823970(160);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  v1 = unk_4D03EC0;
  *(_QWORD *)(result + 16) = 0;
  *(_DWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = -1;
  *(_QWORD *)(result + 80) = 0;
  *(_QWORD *)(result + 88) = 0;
  *(_QWORD *)(result + 96) = 0;
  *(_QWORD *)(result + 104) = v1;
  *(_QWORD *)(result + 112) = 0;
  *(_QWORD *)(result + 120) = 0;
  *(_DWORD *)(result + 128) = 0;
  *(_BYTE *)(result + 132) = 0;
  *(_QWORD *)(result + 136) = 0;
  *(_QWORD *)(result + 144) = 0;
  *(_QWORD *)(result + 152) = 0;
  return result;
}
