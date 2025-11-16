// Function: sub_8D07C0
// Address: 0x8d07c0
//
__int64 sub_8D07C0()
{
  __int64 result; // rax

  result = qword_4F60548;
  if ( qword_4F60548 )
    qword_4F60548 = *(_QWORD *)qword_4F60548;
  else
    result = sub_823970(24);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_DWORD *)(result + 16) = 0;
  *(_BYTE *)(result + 20) = 0;
  return result;
}
