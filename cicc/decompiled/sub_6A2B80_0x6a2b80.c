// Function: sub_6A2B80
// Address: 0x6a2b80
//
__int64 __fastcall sub_6A2B80(unsigned int a1)
{
  __int64 v1; // r12
  char v2; // bl
  bool v3; // bl
  __int64 v4; // rdx
  char v5; // al
  char v6; // bl

  v1 = sub_6E2F40(0);
  v2 = *(_BYTE *)(qword_4D03C50 + 19LL);
  *(_BYTE *)(qword_4D03C50 + 19LL) = v2 & 0xDF;
  v3 = (v2 & 0x20) != 0;
  sub_69ED20(*(_QWORD *)(v1 + 24) + 8LL, 0, 0, a1);
  v4 = qword_4D03C50;
  v5 = *(_BYTE *)(qword_4D03C50 + 19LL);
  if ( (v5 & 0x20) != 0 )
  {
    *(_BYTE *)(v1 + 9) |= 0x40u;
    v5 = *(_BYTE *)(v4 + 19);
  }
  v6 = v5 & 0xDF | (32 * ((v5 & 0x20) != 0 || v3));
  *(_BYTE *)(v4 + 19) = v6;
  return v1;
}
