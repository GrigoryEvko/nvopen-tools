// Function: sub_85B260
// Address: 0x85b260
//
__int64 sub_85B260()
{
  __int64 result; // rax

  result = qword_4F5FD40;
  if ( qword_4F5FD40 )
    qword_4F5FD40 = *(_QWORD *)qword_4F5FD40;
  else
    result = sub_823970(56);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_DWORD *)(result + 40) = 0;
  *(_WORD *)(result + 44) = 0;
  *(_DWORD *)(result + 48) = 0;
  *(_BYTE *)(result + 52) = 0;
  return result;
}
