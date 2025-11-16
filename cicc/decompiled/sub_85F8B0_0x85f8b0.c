// Function: sub_85F8B0
// Address: 0x85f8b0
//
__int64 __fastcall sub_85F8B0(char a1)
{
  __int64 v2; // r12
  __int64 result; // rax
  char v4; // di

  v2 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  result = qword_4F5FD00;
  if ( qword_4F5FD00 )
    qword_4F5FD00 = *(_QWORD *)qword_4F5FD00;
  else
    result = sub_823970(16);
  *(_BYTE *)(result + 8) = (*(_BYTE *)(v2 + 9) >> 1) & 7;
  *(_BYTE *)(result + 9) = (*(_BYTE *)(v2 + 9) & 0x10) != 0;
  *(_QWORD *)result = qword_4F5FD08;
  v4 = *(_BYTE *)(v2 + 9);
  qword_4F5FD08 = result;
  *(_BYTE *)(v2 + 9) = v4 & 0xE1 | (2 * a1) & 0xE | 0x10;
  return result;
}
