// Function: sub_67BB20
// Address: 0x67bb20
//
__int64 __fastcall sub_67BB20(int a1)
{
  __int64 result; // rax

  result = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    result = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(result + 8) = 0;
  *(_DWORD *)result = a1;
  switch ( a1 )
  {
    case 2:
      *(_QWORD *)(result + 16) = unk_4F077C8;
      break;
    case 4:
      *(_QWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 24) = 0xFFFFFFFFLL;
      *(_WORD *)(result + 32) = 0;
      *(_BYTE *)(result + 34) = 0;
      break;
    case 7:
      *(_BYTE *)(result + 16) = 0;
      *(_QWORD *)(result + 24) = 0;
      break;
    default:
      *(_QWORD *)(result + 16) = 0;
      break;
  }
  return result;
}
