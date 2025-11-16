// Function: sub_7F6C60
// Address: 0x7f6c60
//
void __fastcall sub_7F6C60(__int64 a1, int a2, __int64 a3)
{
  int v4; // edx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // r14

  *(_DWORD *)(a3 + 88) = dword_4F07270[0];
  sub_7296B0(a2);
  v4 = dword_4F04C58;
  dword_4F04C58 = -1;
  *(_DWORD *)(a3 + 92) = v4;
  v5 = qword_4F04C50;
  qword_4F04C50 = a1;
  *(_QWORD *)(a3 + 96) = v5;
  LOBYTE(v5) = dword_4D03EB8[0];
  dword_4D03EB8[0] = 0;
  *(_BYTE *)(a3 + 104) = v5;
  *(_QWORD *)(a3 + 112) = qword_4D03F60;
  v6 = unk_4D03F40;
  unk_4D03F40 = 0;
  *(_QWORD *)(a3 + 120) = v6;
  v7 = unk_4D03EB0;
  unk_4D03EB0 = 0;
  *(_QWORD *)(a3 + 176) = v7;
  if ( dword_4F077C4 == 2 )
  {
    nullsub_12();
    if ( !*(_QWORD *)(a1 + 88) )
    {
      v8 = qword_4F06BC0;
      qword_4F06BC0 = *(_QWORD *)(qword_4F07288 + 88);
      sub_733780(0x17u, a1, 0, 1, 0);
      qword_4F06BC0 = v8;
    }
  }
  sub_7E18E0(a3, a1, 0);
  *(_BYTE *)(a3 + 26) = 1;
  sub_7E9AD0();
}
