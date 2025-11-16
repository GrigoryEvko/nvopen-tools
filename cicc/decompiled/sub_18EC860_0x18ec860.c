// Function: sub_18EC860
// Address: 0x18ec860
//
char __fastcall sub_18EC860(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v5; // r15d
  unsigned int i; // ebx

  LODWORD(v3) = *(unsigned __int8 *)(a3 + 16) - 60;
  if ( (unsigned int)v3 > 0xC )
  {
    v5 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
    if ( v5 )
    {
      for ( i = 0; i != v5; ++i )
      {
        while ( 1 )
        {
          LOBYTE(v3) = sub_1AED280(a3, i);
          if ( (_BYTE)v3 )
            break;
          if ( *(_BYTE *)(a3 + 16) == 78 )
          {
            v3 = *(_QWORD *)(a3 - 24);
            if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
              break;
          }
          if ( v5 == ++i )
            return v3;
        }
        LOBYTE(v3) = sub_18EC750(a1, a2, a3, i);
      }
    }
  }
  return v3;
}
