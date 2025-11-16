// Function: sub_15B1130
// Address: 0x15b1130
//
__int64 __fastcall sub_15B1130(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rdx
  unsigned __int8 v3; // al
  __int64 v5; // rcx

  v2 = *(unsigned __int8 **)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)));
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *v2;
      if ( *v2 > 0xEu )
        break;
      if ( v3 > 0xAu )
      {
        v5 = *((_QWORD *)v2 + 4);
        if ( v5 )
          goto LABEL_10;
      }
      if ( v3 == 12 )
      {
        v2 = *(unsigned __int8 **)&v2[8 * (3LL - *((unsigned int *)v2 + 2))];
        if ( v2 )
          continue;
      }
      goto LABEL_6;
    }
    if ( (unsigned __int8)(v3 - 32) <= 1u )
    {
      v5 = *((_QWORD *)v2 + 4);
      if ( v5 )
      {
LABEL_10:
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v5;
        return a1;
      }
    }
  }
LABEL_6:
  *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
