// Function: sub_37F4460
// Address: 0x37f4460
//
__int64 __fastcall sub_37F4460(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int16 v4; // ax

  for ( ; a3 != a2; a2 = *(_QWORD *)(a2 + 8) )
  {
    v4 = *(_WORD *)(a2 + 68);
    if ( (unsigned __int16)(v4 - 14) > 4u && (a4 != 1 || v4 != 24) )
      break;
  }
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a3;
  *(_BYTE *)(a1 + 40) = a4;
  return a1;
}
