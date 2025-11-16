// Function: sub_877830
// Address: 0x877830
//
_QWORD *sub_877830()
{
  const char *v0; // rsi
  size_t v1; // r13
  __int64 v2; // rbx
  char *v3; // rax
  char *v4; // r12
  _QWORD *result; // rax

  v0 = qword_4F06410;
  if ( qword_4F06410 )
  {
    v1 = qword_4F06408 - (_QWORD)qword_4F06410 + 2LL;
    v2 = qword_4F06408 - (_QWORD)qword_4F06410 + 1LL;
  }
  else
  {
    v2 = 0;
    v0 = byte_3F871B3;
    v1 = 1;
  }
  v3 = sub_7248C0(dword_4F073B8[0], v0, v1);
  v3[v2] = 0;
  v4 = v3;
  sub_724C70((__int64)xmmword_4F06220, 2);
  unk_4F062D0 = v1;
  unk_4F062D8 = v4;
  result = sub_73C8D0(0, v1);
  unk_4F062A0 = result;
  return result;
}
