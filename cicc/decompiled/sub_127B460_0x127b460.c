// Function: sub_127B460
// Address: 0x127b460
//
__int64 __fastcall sub_127B460(__int64 a1)
{
  __int64 i; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdx

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = 1;
  if ( (*(_BYTE *)(i + 179) & 0x20) != 0 )
    return 1;
  v3 = *(_QWORD *)(a1 + 160);
  if ( v3 )
  {
    while ( (*(_BYTE *)(v3 + 144) & 1) == 0 )
    {
      v3 = *(_QWORD *)(v3 + 112);
      if ( !v3 )
        goto LABEL_9;
    }
    return 1;
  }
LABEL_9:
  LOBYTE(v2) = *(_DWORD *)(i + 184) != 0;
  return v2;
}
