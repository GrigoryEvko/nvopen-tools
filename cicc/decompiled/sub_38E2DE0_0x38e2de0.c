// Function: sub_38E2DE0
// Address: 0x38e2de0
//
char __fastcall sub_38E2DE0(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v4; // rax

  while ( 2 )
  {
    switch ( *(_DWORD *)a2 )
    {
      case 0:
        result = sub_38E2DE0(a1, *(_QWORD *)(a2 + 24));
        if ( result )
          return result;
        a2 = *(_QWORD *)(a2 + 32);
        continue;
      case 1:
      case 4:
        return 0;
      case 2:
        v4 = *(_QWORD *)(a2 + 24);
        if ( (*(_BYTE *)(v4 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v4 + 8) |= 4u;
          a2 = *(_QWORD *)(v4 + 24);
          continue;
        }
        return a1 == v4;
      case 3:
        a2 = *(_QWORD *)(a2 + 24);
        continue;
    }
  }
}
