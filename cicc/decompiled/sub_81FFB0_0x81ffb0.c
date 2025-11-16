// Function: sub_81FFB0
// Address: 0x81ffb0
//
int sub_81FFB0()
{
  __int64 *v0; // r12
  _QWORD *v1; // rax
  __int64 i; // rbx
  __int64 j; // rbx

  v0 = *(__int64 **)(qword_4F04C68[0] + 24LL);
  if ( !v0 )
    v0 = (__int64 *)(qword_4F04C68[0] + 32LL);
  v1 = &qword_4D04970;
  for ( i = qword_4D04970; i; i = *(_QWORD *)(i + 16) )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(i + 80) == 1 )
      {
        v1 = *(_QWORD **)(i + 88);
        if ( v1[2] )
        {
          if ( (*(_BYTE *)v1 & 4) == 0 )
          {
            v1 = &qword_4D03BC8;
            if ( qword_4D03BC8 != i )
            {
              v1 = &qword_4D03BC0;
              if ( qword_4D03BC0 != i )
              {
                v1 = &qword_4D03BA8;
                if ( qword_4D03BA8 != i )
                  break;
              }
            }
          }
        }
      }
      i = *(_QWORD *)(i + 16);
      if ( !i )
        goto LABEL_13;
    }
    sub_819BD0((_QWORD *)i);
    LODWORD(v1) = fprintf(qword_4D04928, "%s\n", (const char *)qword_4F06C50);
  }
LABEL_13:
  for ( j = *v0; j; j = *(_QWORD *)(j + 16) )
  {
    while ( *(_BYTE *)(j + 80) != 1 )
    {
      j = *(_QWORD *)(j + 16);
      if ( !j )
        return (int)v1;
    }
    sub_819BD0((_QWORD *)j);
    LODWORD(v1) = fprintf(qword_4D04928, "%s\n", (const char *)qword_4F06C50);
  }
  return (int)v1;
}
