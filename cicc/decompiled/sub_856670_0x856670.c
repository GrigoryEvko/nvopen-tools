// Function: sub_856670
// Address: 0x856670
//
unsigned __int16 *__fastcall sub_856670(
        unsigned __int64 a1,
        unsigned int *a2,
        _QWORD **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  _QWORD **v6; // rcx
  _QWORD *v7; // rax
  unsigned __int16 *result; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rax

  *(_BYTE *)(a1 + 96) = 4;
  if ( (_BYTE)a2 == 6 )
  {
    v10 = qword_4D03CB8;
    if ( qword_4D03CB8 )
    {
      v6 = (_QWORD **)*qword_4D03CB8;
      qword_4D03CB8 = (_QWORD *)*qword_4D03CB8;
      a3 = (_QWORD **)qword_4F5FC00;
      *v10 = qword_4F5FC00;
      qword_4F5FC00 = (__int64)v10;
    }
    else
    {
      a2 = &dword_4F063F8;
      a1 = 2618;
      sub_684B30(0xA3Au, &dword_4F063F8);
    }
  }
  else if ( (_BYTE)a2 == 7 )
  {
    v6 = &qword_4D03CB8;
    v7 = qword_4D03CB8;
    if ( qword_4D03CB8 )
    {
      while ( *v7 )
        v7 = (_QWORD *)*v7;
      a3 = (_QWORD **)qword_4F5FC00;
      *v7 = qword_4F5FC00;
      qword_4F5FC00 = (__int64)qword_4D03CB8;
    }
    qword_4D03CB8 = 0;
  }
  else
  {
    v9 = (_QWORD *)qword_4F5FC00;
    if ( qword_4F5FC00 )
    {
      qword_4F5FC00 = *(_QWORD *)qword_4F5FC00;
    }
    else
    {
      a1 = 16;
      v9 = (_QWORD *)sub_823970(16);
    }
    a3 = &qword_4D03CB8;
    *v9 = 0;
    v9[1] = 0;
    v6 = (_QWORD **)qword_4D03CB8;
    *v9 = qword_4D03CB8;
    qword_4D03CB8 = v9;
  }
  sub_7B8B50(a1, a2, (__int64)a3, (__int64)v6, a5, a6);
  result = word_4F06418;
  if ( word_4F06418[0] != 9 )
    return (unsigned __int16 *)sub_684B30(0xEu, dword_4F07508);
  return result;
}
