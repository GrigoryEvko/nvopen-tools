// Function: sub_819470
// Address: 0x819470
//
void *__fastcall sub_819470(int a1, int a2, _WORD *a3)
{
  _BYTE *v3; // r12
  const char *v4; // rsi

  v3 = a3;
  if ( a1 )
  {
    v3 = a3 + 1;
    *a3 = 1024;
  }
  if ( a2 )
    *v3++ = 32;
  if ( (_DWORD)qword_4F06200 )
  {
    if ( !a1 )
    {
      v3 += 2;
      *((_WORD *)v3 - 1) = 1024;
    }
  }
  else if ( HIDWORD(qword_4F06200) )
  {
    v3 += 2;
    *((_WORD *)v3 - 1) = 1280;
  }
  v4 = qword_4F06410;
  if ( !unk_4F06458
    || word_4F06418[0] != 7
    || !unk_4F07710
    || qword_4F06498 > (unsigned __int64)qword_4F06410
    || qword_4F06490 <= (unsigned __int64)qword_4F06410 )
  {
    return memcpy(v3, v4, unk_4F06400);
  }
  if ( (sub_7B7F70((char *)qword_4F06410) & 8) == 0 )
  {
    v4 = qword_4F06410;
    return memcpy(v3, v4, unk_4F06400);
  }
  return (void *)sub_819070(v3);
}
