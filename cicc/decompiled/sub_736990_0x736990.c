// Function: sub_736990
// Address: 0x736990
//
__int64 __fastcall sub_736990(__int64 a1)
{
  char v1; // al
  __int64 i; // rdx
  __int64 v4; // rdx

LABEL_1:
  v1 = *(_BYTE *)(a1 + 89);
  while ( 1 )
  {
    for ( i = *(_QWORD *)(a1 + 40); (v1 & 4) != 0; i = *(_QWORD *)(a1 + 40) )
    {
      a1 = *(_QWORD *)(i + 32);
      v1 = *(_BYTE *)(a1 + 89);
    }
    if ( !i || *(_BYTE *)(i + 28) != 3 || (v4 = *(_QWORD *)(i + 32)) == 0 )
    {
      if ( (v1 & 1) == 0 )
        return 0;
      a1 = *(_QWORD *)(a1 + 48);
      if ( !a1 )
        return 0;
      goto LABEL_1;
    }
    v1 = *(_BYTE *)(v4 + 89);
    if ( (v1 & 0x40) != 0 )
      return 1;
    if ( !((v1 & 8) != 0 ? *(_QWORD *)(v4 + 24) : *(_QWORD *)(v4 + 8)) )
      return 1;
    a1 = v4;
  }
}
