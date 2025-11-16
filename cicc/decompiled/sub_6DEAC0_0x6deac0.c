// Function: sub_6DEAC0
// Address: 0x6deac0
//
__int64 __fastcall sub_6DEAC0(__int64 a1)
{
  char v1; // al
  __int64 v2; // rax
  unsigned int v3; // r8d

  v1 = *(_BYTE *)(a1 + 24);
  if ( v1 == 1 )
  {
LABEL_5:
    v3 = 0;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 56) - 94) <= 1u )
      return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 56LL) + 144LL) & 4) != 0;
    return v3;
  }
  else
  {
    while ( v1 == 3 )
    {
      v2 = *(_QWORD *)(a1 + 56);
      if ( *(_BYTE *)(v2 + 177) != 5 )
        break;
      a1 = *(_QWORD *)(v2 + 184);
      v1 = *(_BYTE *)(a1 + 24);
      if ( v1 == 1 )
        goto LABEL_5;
    }
    return 0;
  }
}
