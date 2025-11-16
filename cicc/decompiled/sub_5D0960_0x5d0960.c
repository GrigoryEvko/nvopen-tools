// Function: sub_5D0960
// Address: 0x5d0960
//
__int64 __fastcall sub_5D0960(char a1, char a2)
{
  __int64 result; // rax
  char v4; // si
  __int64 v5; // rdx

  result = qword_4CF6E38;
  if ( qword_4CF6E38 )
    qword_4CF6E38 = *(_QWORD *)qword_4CF6E38;
  else
    result = sub_823970(16);
  v4 = *(_BYTE *)(result + 9);
  v5 = qword_4CF6E40;
  *(_BYTE *)(result + 8) = a1;
  *(_QWORD *)result = v5;
  qword_4CF6E40 = result;
  *(_BYTE *)(result + 9) = a2 & 1 | v4 & 0xFE;
  return result;
}
