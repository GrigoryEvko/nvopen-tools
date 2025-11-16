// Function: sub_5D19D0
// Address: 0x5d19d0
//
__int64 __fastcall sub_5D19D0(__int64 a1)
{
  __int64 *v1; // rbx
  char v2; // al

  v1 = *(__int64 **)(a1 + 32);
  if ( v1 )
  {
    while ( 1 )
    {
      v2 = *((_BYTE *)v1 + 10);
      switch ( v2 )
      {
        case 3:
          if ( *(_BYTE *)(v1[5] + 173) == 12 )
            return 1;
          goto LABEL_5;
        case 4:
          if ( (unsigned int)sub_8DBE70(v1[5]) )
            return 1;
          if ( *((_BYTE *)v1 + 10) == 5 )
            goto LABEL_11;
LABEL_5:
          v1 = (__int64 *)*v1;
          if ( !v1 )
            return 0;
          break;
        case 5:
LABEL_11:
          if ( (unsigned int)sub_8DBE70(*(_QWORD *)v1[5]) )
            return 1;
          v1 = (__int64 *)*v1;
          if ( !v1 )
            return 0;
          break;
        default:
          goto LABEL_5;
      }
    }
  }
  return 0;
}
