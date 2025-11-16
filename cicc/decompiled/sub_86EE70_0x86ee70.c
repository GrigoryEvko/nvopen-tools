// Function: sub_86EE70
// Address: 0x86ee70
//
__int64 sub_86EE70()
{
  __int64 v0; // rax
  __int64 v1; // rdx
  __int64 result; // rax

  sub_86EC60(1, 0, 0);
  v0 = 776LL * dword_4F04C64;
  v1 = qword_4F04C68[0] + v0 - 776;
  *(_QWORD *)(qword_4F04C68[0] + v0 + 440) = *(_QWORD *)(v1 + 440);
  result = 176LL * unk_4D03B90;
  *(_QWORD *)(qword_4D03B98 + result + 160) = *(_QWORD *)(qword_4D03B98 + result - 16);
  *(_QWORD *)(v1 + 440) = 0;
  return result;
}
