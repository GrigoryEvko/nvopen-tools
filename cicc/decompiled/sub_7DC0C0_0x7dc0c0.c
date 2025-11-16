// Function: sub_7DC0C0
// Address: 0x7dc0c0
//
__int64 sub_7DC0C0()
{
  __int64 result; // rax
  _QWORD *v1; // rax
  unsigned __int64 v2; // rsi

  result = qword_4F188C0;
  if ( !qword_4F188C0 )
  {
    qword_4F188C0 = sub_7E16B0(10);
    sub_7E1CA0(qword_4F188C0);
    sub_7DC070();
    sub_7E1B70((char *)"tinfo");
    sub_72BA30(unk_4F06870);
    sub_7E1B70("flags");
    v1 = sub_72BA30(unk_4F06870);
    v2 = sub_72D2E0(v1);
    sub_7E1B70("ptr_flags");
    sub_7E1C00(qword_4F188C0, v2);
    return qword_4F188C0;
  }
  return result;
}
