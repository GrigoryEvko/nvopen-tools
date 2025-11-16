// Function: sub_7F51D0
// Address: 0x7f51d0
//
void __fastcall sub_7F51D0(__int64 a1, int a2, int a3, __int64 a4)
{
  char v4; // al
  bool v5; // sf
  int v6; // eax

  while ( 1 )
  {
    if ( !a1 )
    {
      *(_QWORD *)a4 = 0;
      *(_QWORD *)(a4 + 8) = 0;
      *(_DWORD *)(a4 + 16) = a2;
      *(_DWORD *)(a4 + 20) = a3;
      return;
    }
    v4 = *(_BYTE *)(a1 + 171);
    if ( (v4 & 0x20) == 0 )
      break;
    v5 = v4 < 0;
    v6 = a3;
    if ( v5 )
      v6 = a2;
    if ( !v6 )
      break;
    a1 = *(_QWORD *)(a1 + 120);
  }
  *(_QWORD *)a4 = a1;
  *(_QWORD *)(a4 + 8) = 0;
  *(_DWORD *)(a4 + 16) = a2;
  *(_DWORD *)(a4 + 20) = a3;
  if ( *(_BYTE *)(a1 + 173) == 11 )
    *(_QWORD *)(a4 + 8) = *(_QWORD *)(a1 + 184);
}
