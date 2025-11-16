// Function: sub_1FD3B40
// Address: 0x1fd3b40
//
__int64 __fastcall sub_1FD3B40(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  _BYTE *v4; // rdi
  _BYTE *v6; // rax
  _BYTE *v7; // r13

  v4 = a2;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      v6 = v4;
      if ( (*v4 & 4) == 0 && (v4[46] & 8) != 0 )
      {
        do
          v6 = (_BYTE *)*((_QWORD *)v6 + 1);
        while ( (v6[46] & 8) != 0 );
      }
      v7 = (_BYTE *)*((_QWORD *)v6 + 1);
      sub_1E16240((__int64)v4);
      if ( a3 == v7 )
        break;
      v4 = v7;
    }
  }
  return sub_1FD3A30(a1);
}
