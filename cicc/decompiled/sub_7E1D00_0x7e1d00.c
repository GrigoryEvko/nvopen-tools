// Function: sub_7E1D00
// Address: 0x7e1d00
//
__int64 sub_7E1D00()
{
  __int64 result; // rax
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // r13
  _QWORD *v4; // rsi
  __int64 v5[4]; // [rsp-20h] [rbp-20h] BYREF

  result = qword_4F18A00;
  if ( !qword_4F18A00 )
  {
    qword_4F18A00 = (__int64)sub_7E16B0(10);
    sub_7E1CA0(qword_4F18A00);
    v1 = qword_4F18A00;
    v5[0] = 0;
    v2 = sub_7E1C50();
    sub_7E1B70("f", v2, v1, v5);
    v3 = qword_4F18A00;
    qword_4F189F0 = v5[0];
    v4 = sub_72BA30(unk_4F06A60);
    sub_7E1B70("d", (__int64)v4, v3, v5);
    qword_4F189F8 = v5[0];
    sub_7E1C00(qword_4F18A00, (unsigned __int64)v4);
    return qword_4F18A00;
  }
  return result;
}
