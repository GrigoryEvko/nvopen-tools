// Function: sub_301EA30
// Address: 0x301ea30
//
__int64 sub_301EA30()
{
  __int64 result; // rax

  result = sub_22077B0(0x30u);
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)result = &unk_44A4E58;
  *(_DWORD *)(result + 40) = 7056;
  *(_QWORD *)(result + 8) = &unk_44591A0;
  *(_QWORD *)(result + 16) = &unk_445FFE0;
  return result;
}
