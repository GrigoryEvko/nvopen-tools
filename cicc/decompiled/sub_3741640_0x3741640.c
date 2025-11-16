// Function: sub_3741640
// Address: 0x3741640
//
__int64 __fastcall sub_3741640(_QWORD *a1, _BYTE *a2, _BYTE *a3)
{
  _BYTE *v4; // rdi
  _BYTE *v6; // rax
  _BYTE *v7; // r13

  v4 = a2;
  if ( a2 != a3 )
  {
    if ( (_BYTE *)a1[22] == a2 )
      goto LABEL_11;
    while ( (_BYTE *)a1[21] == v4 )
    {
      while ( 1 )
      {
        a1[21] = a3;
        if ( (_BYTE *)a1[20] == v4 )
LABEL_13:
          a1[20] = a3;
LABEL_5:
        if ( !v4 )
          BUG();
        v6 = v4;
        if ( (*v4 & 4) == 0 && (v4[44] & 8) != 0 )
        {
          do
            v6 = (_BYTE *)*((_QWORD *)v6 + 1);
          while ( (v6[44] & 8) != 0 );
        }
        v7 = (_BYTE *)*((_QWORD *)v6 + 1);
        sub_2E88E20((__int64)v4);
        if ( a3 == v7 )
          return sub_3741080((__int64)a1);
        v4 = v7;
        if ( (_BYTE *)a1[22] != v7 )
          break;
LABEL_11:
        a1[22] = a3;
        if ( (_BYTE *)a1[21] != v4 )
          goto LABEL_4;
      }
    }
LABEL_4:
    if ( (_BYTE *)a1[20] == v4 )
      goto LABEL_13;
    goto LABEL_5;
  }
  return sub_3741080((__int64)a1);
}
