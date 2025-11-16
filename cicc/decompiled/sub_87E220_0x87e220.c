// Function: sub_87E220
// Address: 0x87e220
//
__int64 sub_87E220()
{
  __int64 result; // rax
  __int64 v1; // rdx
  __int64 v2; // rdx

  result = sub_823970(32);
  v1 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 240);
  *(_QWORD *)(result + 8) = 0;
  *(_DWORD *)(result + 16) = 0;
  *(_QWORD *)result = v1;
  v2 = dword_4F04C64;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(qword_4F04C68[0] + 776 * v2 + 240) = result;
  return result;
}
