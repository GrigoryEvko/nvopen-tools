// Function: sub_37417A0
// Address: 0x37417a0
//
void __fastcall sub_37417A0(_QWORD *a1, _BYTE *a2)
{
  _BYTE *v3; // rax
  _BYTE *v4; // rsi
  __int64 v5; // rax

  if ( (_BYTE *)a1[20] != a2 )
  {
    if ( a2 )
    {
      v3 = a2;
      if ( (*a2 & 4) == 0 && (a2[44] & 8) != 0 )
      {
        do
          v3 = (_BYTE *)*((_QWORD *)v3 + 1);
        while ( (v3[44] & 8) != 0 );
      }
      v4 = (_BYTE *)*((_QWORD *)v3 + 1);
    }
    else
    {
      v4 = (_BYTE *)sub_2E311E0(*(_QWORD *)(a1[5] + 744LL));
    }
    v5 = a1[5];
    a1[21] = a2;
    a1[20] = a2;
    sub_3741640(a1, v4, *(_BYTE **)(v5 + 752));
  }
}
