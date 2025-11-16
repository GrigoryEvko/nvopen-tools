// Function: sub_1E15D00
// Address: 0x1e15d00
//
bool __fastcall sub_1E15D00(__int64 a1, unsigned int a2, int a3)
{
  __int64 v3; // rax

  while ( 1 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( (*(_QWORD *)(v3 + 8) & a2) != 0 )
      break;
    if ( a3 == 2 && *(_WORD *)v3 != 16 )
      return 0;
LABEL_3:
    if ( (*(_BYTE *)(a1 + 46) & 8) == 0 )
      return a3 == 2;
    a1 = *(_QWORD *)(a1 + 8);
  }
  if ( a3 != 1 )
    goto LABEL_3;
  return 1;
}
