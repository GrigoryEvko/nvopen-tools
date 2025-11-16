// Function: sub_1FD3D60
// Address: 0x1fd3d60
//
void __fastcall sub_1FD3D60(_QWORD *a1, _BYTE *a2)
{
  _BYTE *v3; // rax
  _BYTE *v4; // rsi
  __int64 v5; // rax

  if ( (_BYTE *)a1[18] != a2 )
  {
    if ( a2 )
    {
      v3 = a2;
      if ( (*a2 & 4) == 0 && (a2[46] & 8) != 0 )
      {
        do
          v3 = (_BYTE *)*((_QWORD *)v3 + 1);
        while ( (v3[46] & 8) != 0 );
      }
      v4 = (_BYTE *)*((_QWORD *)v3 + 1);
    }
    else
    {
      v4 = (_BYTE *)sub_1DD5D10(*(_QWORD *)(a1[5] + 784LL));
    }
    v5 = a1[5];
    a1[19] = a2;
    a1[18] = a2;
    sub_1FD3B40((__int64)a1, v4, *(_BYTE **)(v5 + 792));
  }
}
