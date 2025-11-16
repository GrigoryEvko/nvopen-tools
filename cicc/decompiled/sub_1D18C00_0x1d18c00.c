// Function: sub_1D18C00
// Address: 0x1d18c00
//
bool __fastcall sub_1D18C00(__int64 a1, int a2, int a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a1 + 48);
  if ( !v3 )
    return a2 == 0;
  while ( 1 )
  {
    while ( a3 != *(_DWORD *)(v3 + 8) )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        return a2 == 0;
    }
    if ( !a2 )
      break;
    v3 = *(_QWORD *)(v3 + 32);
    --a2;
    if ( !v3 )
      return a2 == 0;
  }
  return 0;
}
