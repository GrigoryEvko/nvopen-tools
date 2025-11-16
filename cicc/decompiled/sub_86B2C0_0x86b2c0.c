// Function: sub_86B2C0
// Address: 0x86b2c0
//
__int64 __fastcall sub_86B2C0(char a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = qword_4F5FD60;
  if ( qword_4F5FD60 )
    qword_4F5FD60 = *(_QWORD *)qword_4F5FD60;
  else
    result = sub_823970(80);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  v2 = *(_QWORD *)dword_4F07508;
  *(_QWORD *)(result + 16) = 0;
  *(_BYTE *)(result + 32) = a1;
  *(_QWORD *)(result + 24) = v2;
  switch ( a1 )
  {
    case 1:
      *(_BYTE *)(result + 56) &= 0xFCu;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      break;
    case 2:
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      break;
    case 3:
    case 5:
      *(_QWORD *)(result + 40) = 0;
      break;
    case 4:
      return result;
    default:
      *(_WORD *)(result + 72) &= 0xF800u;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_QWORD *)(result + 56) = 0;
      *(_QWORD *)(result + 64) = 0;
      break;
  }
  return result;
}
