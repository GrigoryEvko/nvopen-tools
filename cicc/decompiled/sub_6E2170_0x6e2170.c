// Function: sub_6E2170
// Address: 0x6e2170
//
void __fastcall sub_6E2170(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  int v3; // eax
  char v4; // cl
  char v5; // si

  if ( a1 )
  {
    v1 = qword_4D03C50;
    if ( qword_4D03C50 )
    {
      if ( (*(_BYTE *)(a1 + 19) & 8) != 0 )
        goto LABEL_7;
      v2 = *(_QWORD *)(qword_4D03C50 + 144LL);
      if ( v2 )
      {
        if ( v2 != *(_QWORD *)(a1 + 144) )
          return;
LABEL_7:
        sub_6DE9B0(1, a1, qword_4D03C50);
        return;
      }
      v3 = *(_DWORD *)(a1 + 56);
      if ( v3 != -1 && v3 == *(_DWORD *)(qword_4D03C50 + 56LL) )
      {
        *(_BYTE *)(qword_4D03C50 + 17LL) = *(_BYTE *)(a1 + 17) & 0x40 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xBF;
        *(_QWORD *)(v1 + 112) = *(_QWORD *)(a1 + 112);
        *(_QWORD *)(v1 + 120) = *(_QWORD *)(a1 + 120);
        v4 = *(_BYTE *)(v1 + 21);
        *(_BYTE *)(v1 + 20) = (*(_BYTE *)(v1 + 20) | *(_BYTE *)(a1 + 20)) & 0x80 | *(_BYTE *)(v1 + 20) & 0x7F;
        v5 = (v4 | *(_BYTE *)(a1 + 21)) & 8;
        *(_BYTE *)(v1 + 21) = v5 | v4 & 0xF7;
        *(_BYTE *)(v1 + 21) = v5 & 0x10 | v4 & 0x10 | *(_BYTE *)(a1 + 21) & 0x10 | v5 & 0xEF | v4 & 0xE7;
      }
    }
  }
}
