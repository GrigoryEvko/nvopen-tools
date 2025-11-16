// Function: sub_733B20
// Address: 0x733b20
//
void __fastcall sub_733B20(_QWORD *a1)
{
  _QWORD *v1; // rcx
  _QWORD *v2; // rax
  __int64 v3; // rsi
  _QWORD *v4; // rdx
  __int64 i; // rax
  __int64 v6; // rax
  char v7; // dl

  v1 = (_QWORD *)a1[3];
  if ( v1 )
  {
    v2 = (_QWORD *)v1[3];
    v3 = a1[4];
    if ( v2 == a1 )
    {
      v1[3] = v3;
      if ( !v3 )
      {
        v6 = v1[5];
        if ( v6 )
        {
          v7 = *(_BYTE *)(v6 + 50);
          if ( (v7 & 4) != 0 && *(_QWORD **)(v6 + 104) == v1 )
          {
            *(_QWORD *)(v6 + 104) = 0;
            *(_BYTE *)(v6 + 50) = v7 & 0xFB;
            v1[5] = *(_QWORD *)(v6 + 32);
          }
        }
      }
    }
    else
    {
      do
      {
        v4 = v2;
        v2 = (_QWORD *)v2[4];
      }
      while ( v2 != a1 );
      v4[4] = v3;
    }
    for ( i = v1[6]; i; i = *(_QWORD *)(i + 56) )
    {
      while ( *(_QWORD **)(i + 40) != a1 )
      {
        i = *(_QWORD *)(i + 56);
        if ( !i )
          goto LABEL_10;
      }
      *(_QWORD *)(i + 40) = a1[4];
    }
LABEL_10:
    a1[4] = 0;
    a1[3] = 0;
  }
  if ( a1[10] )
  {
    sub_7F93F0();
    a1[10] = 0;
  }
}
