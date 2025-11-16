// Function: sub_705870
// Address: 0x705870
//
__int64 __fastcall sub_705870(int a1)
{
  const char *v2; // r12
  size_t v3; // rax
  char *v4; // rax
  unsigned int v5; // edi

  if ( unk_4D03FE8 )
    unk_4F07280 = 0;
  v2 = qword_4D03FE0;
  v3 = strlen(qword_4D03FE0);
  v4 = (char *)sub_7279A0(v3 + 1);
  v5 = (unsigned int)strcpy(v4, v2);
  sub_7B2160(v5, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  *(_QWORD *)(unk_4D03FF0 + 176LL) = *(_QWORD *)(unk_4F064B0 + 64LL);
  if ( !(unk_4D03C90 | a1) )
  {
    unk_4F04D90 = unk_4F076D0;
    unk_4F04D88 = 1;
    sub_7B22D0();
  }
  return sub_7B2B10(1, 0);
}
