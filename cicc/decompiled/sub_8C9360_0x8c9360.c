// Function: sub_8C9360
// Address: 0x8c9360
//
void __fastcall sub_8C9360(__int64 a1)
{
  _QWORD *v1; // rbx
  char v2; // al

  v1 = (_QWORD *)qword_4F60240;
  if ( qword_4F60240 )
  {
    do
    {
      while ( v1[1] != a1 )
      {
        v1 = (_QWORD *)*v1;
        if ( !v1 )
          return;
      }
      v2 = *(_BYTE *)(a1 + 80);
      if ( (unsigned __int8)(v2 - 4) <= 1u )
      {
        sub_8CCC20(a1);
      }
      else if ( (unsigned __int8)(v2 - 10) <= 1u || v2 == 17 )
      {
        sub_8CC1D0(a1);
      }
      else if ( v2 == 3 )
      {
        sub_8C9210(a1);
      }
      v1[1] = 0;
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
}
