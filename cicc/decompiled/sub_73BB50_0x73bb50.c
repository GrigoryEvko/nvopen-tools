// Function: sub_73BB50
// Address: 0x73bb50
//
_QWORD *__fastcall sub_73BB50(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v3; // r13
  unsigned __int8 *v4; // r14
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  int v7[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_7296C0(v7);
  if ( *(_BYTE *)(a1 + 24) == 10 )
  {
    v3 = qword_4F06BC0;
    qword_4F06BC0 = *(_QWORD *)(unk_4F07288 + 88LL);
    sub_733780(0, 0, 0, **(_BYTE **)(a1 + 64), 0);
    v4 = (unsigned __int8 *)qword_4F06BC0;
    v5 = sub_73B8B0(*(const __m128i **)(a1 + 56), 0);
    v1 = v5;
    if ( v4 )
    {
      v6 = sub_7335F0(v5, v4);
      qword_4F06BC0 = v3;
      v1 = v6;
    }
  }
  else
  {
    v1 = sub_73B8B0((const __m128i *)a1, 0);
  }
  sub_729730(v7[0]);
  return v1;
}
