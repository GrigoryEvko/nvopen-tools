// Function: sub_8193C0
// Address: 0x8193c0
//
__int64 __fastcall sub_8193C0(int a1, int a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  if ( unk_4F06458
    && word_4F06418[0] == 7
    && unk_4F07710
    && qword_4F06498 <= (unsigned __int64)qword_4F06410
    && (unsigned __int64)qword_4F06410 < qword_4F06490
    && (sub_7B7F70((char *)qword_4F06410) & 8) != 0 )
  {
    v2 = sub_819070(0);
  }
  else
  {
    v2 = unk_4F06400;
  }
  result = v2 - ((a2 == 0) - 1LL);
  if ( a1 )
    result += 2;
  if ( HIDWORD(qword_4F06200) )
    result += 2;
  return result;
}
