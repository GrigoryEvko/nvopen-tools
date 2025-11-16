// Function: sub_6BB6B0
// Address: 0x6bb6b0
//
unsigned __int16 *__fastcall sub_6BB6B0(__int64 a1)
{
  unsigned __int16 *result; // rax
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  _BYTE v5[192]; // [rsp+0h] [rbp-C0h] BYREF

  sub_6E1BE0(a1);
  result = word_4F06418;
  if ( word_4F06418[0] != 28 )
  {
    sub_6E1E00(4, v5, 1, 0);
    v2 = qword_4F061C8;
    v3 = qword_4D03C50;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(v3 + 40);
    ++*(_BYTE *)(v2 + 75);
    do
      sub_6BB600(a1, 1u);
    while ( (unsigned int)sub_7BE800(67) );
    v4 = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    --*(_BYTE *)(v4 + 36);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    return (unsigned __int16 *)sub_6E2B30(67, 1);
  }
  return result;
}
