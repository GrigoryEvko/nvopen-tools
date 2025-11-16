// Function: sub_5CEEC0
// Address: 0x5ceec0
//
void __fastcall sub_5CEEC0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // r12
  char v6; // al

  if ( a2 )
  {
    v4 = a2;
    do
    {
      while ( 1 )
      {
        v5 = (__int64)v4;
        v4 = (_QWORD *)*v4;
        v6 = *(_BYTE *)(v5 + 11);
        if ( (*(_BYTE *)(v5 + 9) == 2 || (v6 & 0x10) != 0) && (v6 & 2) != 0 )
          break;
        if ( !v4 )
          return;
      }
      *(_QWORD *)(v5 + 48) = a3;
      *a1 = sub_5CD370(v5, *a1, 6u);
      *(_QWORD *)(v5 + 48) = 0;
    }
    while ( v4 );
  }
}
