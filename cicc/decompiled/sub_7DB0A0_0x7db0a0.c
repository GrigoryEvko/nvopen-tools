// Function: sub_7DB0A0
// Address: 0x7db0a0
//
__int64 __fastcall sub_7DB0A0(__int64 a1, unsigned int a2, _QWORD *a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r12
  _BYTE *v6; // rax
  _BYTE *v7; // rsi

  v4 = sub_7259C0(8);
  v4[20] = a1;
  v4[22] = 0;
  v5 = sub_7E9300(v4, a2);
  v6 = sub_724D50(10);
  *a3 = v6;
  if ( a2 )
    v7 = (_BYTE *)qword_4F04C50;
  else
    v7 = *(_BYTE **)(unk_4D03F68 + 8LL);
  sub_7333B0(v5, v7, 1, (__int64)v6, 0);
  return v5;
}
