// Function: sub_8555D0
// Address: 0x8555d0
//
unsigned int *__fastcall sub_8555D0(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  int v6; // r15d
  int v7; // r14d
  int v8; // r13d
  unsigned __int8 v9; // al
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v19; // [rsp+14h] [rbp-1ACh]
  int v20; // [rsp+18h] [rbp-1A8h]
  int v21; // [rsp+1Ch] [rbp-1A4h]
  _BYTE v22[352]; // [rsp+20h] [rbp-1A0h] BYREF
  int v23; // [rsp+180h] [rbp-40h]
  __int16 v24; // [rsp+184h] [rbp-3Ch]

  v3 = a1 + 16;
  v4 = a1 + 16;
  sub_7AE360(v4);
  v6 = dword_4D03D1C;
  v7 = unk_4D03D10;
  v8 = dword_4D03D08;
  v21 = unk_4D03D20;
  v20 = unk_4D03D0C;
  v19 = dword_4D03D18;
  dword_4D03D18 = 1;
  v9 = *(_BYTE *)(a2 + 17);
  dword_4D03D1C = (v9 & 0x20) != 0;
  unk_4D03D10 = 1;
  unk_4D03D20 = v9 >> 7;
  unk_4D03D0C = (v9 & 0x40) != 0;
  dword_4D03D08 = (v9 & 0x40) != 0;
  sub_7B8B50(v4, (unsigned int *)&unk_4D03D20, (v9 & 0x40) != 0, (__int64)&dword_4D03D08, v10, v11);
  if ( a3 )
  {
    memset(v22, 0, sizeof(v22));
    *(_WORD *)&v22[9] = 257;
    v22[28] = 1;
    v24 = 0;
    v23 = 0;
    sub_7C6880(v3, (__int64)v22, 257, 0, v12, v13);
  }
  else
  {
    while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
    {
      sub_7AE360(v3);
      sub_7B8B50(v3, (unsigned int *)&unk_4D03D20, v14, v15, v16, v17);
    }
  }
  sub_7AE210(v3);
  dword_4D03D1C = v6;
  unk_4D03D10 = v7;
  dword_4D03D08 = v8;
  unk_4D03D20 = v21;
  unk_4D03D0C = v20;
  dword_4D03D18 = v19;
  return &dword_4D03D18;
}
