// Function: sub_88DB10
// Address: 0x88db10
//
__int64 __fastcall sub_88DB10(__int64 *a1)
{
  __int64 *v1; // rbx
  char v2; // al
  __int64 i; // rax
  char v4; // al
  __int64 v6; // rax
  __int64 v7; // rdx

  if ( a1 )
  {
    v1 = a1;
    while ( 1 )
    {
      v2 = *((_BYTE *)v1 + 8);
      if ( !v2 )
      {
        for ( i = v1[4]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
LABEL_9:
        v4 = (*(_BYTE *)(i + 88) >> 4) & 7;
LABEL_10:
        if ( v4 == 1 )
          return 1;
        goto LABEL_5;
      }
      if ( v2 != 1 )
        break;
      if ( (v1[3] & 1) != 0 )
      {
LABEL_5:
        v1 = (__int64 *)*v1;
        if ( !v1 )
          return 0;
      }
      else
      {
        i = sub_72A940(v1[4]);
        if ( i )
          goto LABEL_9;
        v1 = (__int64 *)*v1;
        if ( !v1 )
          return 0;
      }
    }
    if ( v2 == 2 )
    {
      v6 = v1[4];
      v7 = *(_QWORD *)(v6 + 168);
      if ( v7 )
      {
        switch ( *(_BYTE *)(v6 + 120) )
        {
          case 1:
          case 6:
          case 7:
            v4 = (*(_BYTE *)(v7 + 265) >> 2) & 7;
            goto LABEL_10;
          case 2:
          case 4:
            v4 = (*(_BYTE *)(*(_QWORD *)(v7 + 176) + 88LL) >> 4) & 7;
            goto LABEL_10;
          case 3:
          case 5:
            i = *(_QWORD *)(v7 + 192);
            goto LABEL_9;
          default:
            goto LABEL_5;
        }
      }
    }
    goto LABEL_5;
  }
  return 0;
}
