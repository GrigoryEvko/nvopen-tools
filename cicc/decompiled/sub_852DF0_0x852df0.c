// Function: sub_852DF0
// Address: 0x852df0
//
__int64 sub_852DF0()
{
  __int64 v0; // rdx
  __int64 result; // rax

  dword_4D03CB0[0] = 0;
  unk_4D03C84 = 0;
  v0 = qword_4F07280;
  result = qword_4F5F880;
  *(_QWORD *)(qword_4F07280 + 40LL) = qword_4F5F880;
  *(_QWORD *)(v0 + 48) = result;
  *(_BYTE *)(result + 73) |= 1u;
  *(_DWORD *)(v0 + 24) = 1;
  *(_DWORD *)(result + 28) = dword_4F5F878;
  return result;
}
