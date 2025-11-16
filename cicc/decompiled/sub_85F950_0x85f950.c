// Function: sub_85F950
// Address: 0x85f950
//
__int64 sub_85F950()
{
  __int64 v0; // rcx
  __int64 v1; // rdx
  char v2; // al
  __int64 result; // rax

  v0 = qword_4F5FD08;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v2 = (2 * (*(_BYTE *)(qword_4F5FD08 + 8) & 7)) | *(_BYTE *)(v1 + 9) & 0xF1;
  *(_BYTE *)(v1 + 9) = v2;
  *(_BYTE *)(v1 + 9) = (16 * (*(_BYTE *)(v0 + 9) & 1)) | v2 & 0xEF;
  qword_4F5FD08 = *(_QWORD *)v0;
  result = qword_4F5FD00;
  *(_QWORD *)v0 = qword_4F5FD00;
  qword_4F5FD00 = v0;
  return result;
}
