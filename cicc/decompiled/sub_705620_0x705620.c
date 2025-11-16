// Function: sub_705620
// Address: 0x705620
//
_QWORD *__fastcall sub_705620(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int16 v5; // ax

  if ( word_4F06418[0] != 55 )
  {
    v2 = 0;
    sub_6851D0(0x35u);
    return v2;
  }
  sub_7BDAB0(a1);
  if ( word_4F06418[0] == 1 )
  {
    v1 = 0;
    v2 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v3 = sub_64E550(0, 0);
        if ( v2 )
          break;
        v1 = (_QWORD *)sub_7263F0();
        v2 = v1;
        v5 = word_4F06418[0];
        v1[1] = v3;
        if ( v5 != 67 )
          goto LABEL_6;
LABEL_11:
        sub_7BDAB0(a1);
        if ( word_4F06418[0] != 1 )
        {
          sub_6851D0(0x9ACu);
          v5 = word_4F06418[0];
          goto LABEL_6;
        }
      }
      v4 = sub_7263F0();
      *v1 = v4;
      v1 = (_QWORD *)v4;
      v5 = word_4F06418[0];
      v1[1] = v3;
      if ( v5 == 67 )
        goto LABEL_11;
LABEL_6:
      if ( v5 != 1 )
      {
        if ( v5 != 28 )
          goto LABEL_8;
        return v2;
      }
    }
  }
  if ( word_4F06418[0] == 28 )
  {
    v2 = 0;
    sub_6851C0(0x9ACu, &dword_4F063F8);
  }
  else
  {
    v2 = 0;
LABEL_8:
    sub_6851D0(0x12u);
  }
  return v2;
}
