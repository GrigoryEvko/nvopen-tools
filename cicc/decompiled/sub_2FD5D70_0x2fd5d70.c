// Function: sub_2FD5D70
// Address: 0x2fd5d70
//
bool __fastcall sub_2FD5D70(__int64 a1)
{
  __int64 v1; // rax
  __int64 i; // rcx

  v1 = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != v1; v1 = *(_QWORD *)(v1 + 8) )
  {
    if ( *(_WORD *)(v1 + 68) == 68 || !*(_WORD *)(v1 + 68) )
      break;
    if ( (*(_BYTE *)v1 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v1 + 44) & 8) != 0 )
        v1 = *(_QWORD *)(v1 + 8);
    }
  }
  return i != v1;
}
