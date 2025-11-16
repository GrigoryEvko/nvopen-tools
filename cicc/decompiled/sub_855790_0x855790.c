// Function: sub_855790
// Address: 0x855790
//
_QWORD *__fastcall sub_855790(_DWORD *a1, unsigned int *a2)
{
  int v2; // r15d
  int v3; // r14d
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  bool v8; // zf
  int v10; // [rsp+Ch] [rbp-44h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = unk_4D03D20;
  v3 = dword_4D03D1C;
  v4 = sub_724DC0();
  unk_4D03D20 = 0;
  v11[0] = (__int64)v4;
  v5 = (unsigned int)dword_4D03CE4;
  dword_4D03D1C = 1;
  dword_4D03CE4 = 0;
  v10 = v5;
  unk_4D03D04 = 1;
  sub_7B8B50((unsigned __int64)a1, a2, (__int64)&dword_4D03CE4, v5, v6, v7);
  sub_6B9C00(v11[0]);
  v8 = *(_BYTE *)(v11[0] + 173) == 0;
  unk_4D03D04 = 0;
  dword_4D03CE4 = v10;
  if ( v8 )
  {
    *a1 = 0;
    dword_4D03CE0 = 1;
  }
  else
  {
    *a1 = sub_6210B0(v11[0], 0) != 0;
  }
  unk_4D03D20 = v2;
  dword_4D03D1C = v3;
  return sub_724E30((__int64)v11);
}
