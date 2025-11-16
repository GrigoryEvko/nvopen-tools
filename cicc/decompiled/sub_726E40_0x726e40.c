// Function: sub_726E40
// Address: 0x726e40
//
void __fastcall sub_726E40(__int64 a1, char a2, __int64 a3)
{
  *(_BYTE *)(a1 + 28) = a2;
  switch ( a2 )
  {
    case 0:
    case 8:
      return;
    case 1:
    case 2:
    case 3:
    case 6:
    case 15:
    case 16:
      *(_QWORD *)(a1 + 32) = 0;
      break;
    case 17:
      *(_QWORD *)(a1 + 32) = a3;
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      break;
    default:
      sub_721090();
  }
}
