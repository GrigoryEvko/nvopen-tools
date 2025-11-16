// Function: sub_6E1990
// Address: 0x6e1990
//
void __fastcall sub_6E1990(_QWORD *a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rbx
  unsigned __int8 v3; // al

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (__int64)v1;
      v1 = (_QWORD *)*v1;
      if ( (*(_BYTE *)(v2 + 9) & 0x20) != 0 )
        continue;
      v3 = *(_BYTE *)(v2 + 8);
      if ( v3 != 2 )
      {
        if ( v3 <= 2u )
        {
          a1 = *(_QWORD **)(v2 + 24);
          if ( v3 )
            sub_6E1990(a1);
          else
            sub_6E1940(a1);
          *(_QWORD *)(v2 + 24) = 0;
          goto LABEL_4;
        }
        if ( v3 != 3 )
          sub_721090(a1);
      }
      *(_QWORD *)(v2 + 24) = 0;
LABEL_4:
      *(_QWORD *)v2 = qword_4D03A80;
      qword_4D03A80 = v2;
    }
    while ( v1 );
  }
}
