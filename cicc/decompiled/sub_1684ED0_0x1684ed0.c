// Function: sub_1684ED0
// Address: 0x1684ed0
//
_QWORD *sub_1684ED0()
{
  _QWORD *result; // rax
  __int64 v1; // rdi
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // r12

  result = &unk_4CD28E0;
  if ( unk_4CD28E0 )
  {
    v1 = qword_4F9F338;
    qword_4F9F338 = 0;
    if ( v1 )
      ((void (*)(void))sub_1687040)();
    if ( qword_4F9F358 )
    {
      sub_1684B50(&qword_4F9F360);
      if ( qword_4F9F358 )
      {
        sub_1688C60(qword_4F9F358, 1);
        qword_4F9F358 = 0;
        dword_4F9F350 = 0;
      }
      j__pthread_mutex_unlock(qword_4F9F360);
    }
    result = (_QWORD *)sub_1689050();
    v2 = result[12];
    if ( v2 )
    {
      v3 = sub_1689050();
      v4 = qword_4F9F328;
      *(_QWORD *)(v3 + 96) = 0;
      sub_1687E40(v4, v2);
      result = (_QWORD *)sub_1687040(v2);
    }
    if ( qword_4F9F328 )
    {
      if ( !(unsigned __int8)sub_1688080() )
        sub_16876E0(qword_4F9F328, sub_1684B40, 0);
      result = (_QWORD *)sub_1687510(qword_4F9F328);
      qword_4F9F328 = 0;
    }
    if ( qword_4F9F360 )
    {
      sub_1688E30();
      if ( qword_4F9F360 )
      {
        v5 = sub_1683C60(0);
        sub_1688DA0(qword_4F9F360);
        qword_4F9F360 = 0;
        sub_1683C60(v5);
      }
      result = (_QWORD *)sub_1688E70();
    }
    if ( qword_4F9F330 )
    {
      sub_1688E30();
      if ( qword_4F9F330 )
      {
        v6 = sub_1683C60(0);
        sub_1688DA0(qword_4F9F330);
        qword_4F9F330 = 0;
        sub_1683C60(v6);
      }
      return (_QWORD *)sub_1688E70();
    }
  }
  return result;
}
