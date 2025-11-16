// Function: sub_10391D0
// Address: 0x10391d0
//
__int64 __fastcall sub_10391D0(__int64 a1, char a2)
{
  __int64 result; // rax

  result = a1;
  switch ( a2 )
  {
    case 2:
      strcpy((char *)(a1 + 16), "cold");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 4;
      break;
    case 4:
      *(_BYTE *)(a1 + 18) = 116;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 28520;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      break;
    case 1:
      *(_BYTE *)(a1 + 22) = 100;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1668575086;
      *(_WORD *)(a1 + 20) = 27759;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      break;
    default:
      BUG();
  }
  return result;
}
