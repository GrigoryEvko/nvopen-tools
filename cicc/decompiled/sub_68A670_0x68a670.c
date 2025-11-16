// Function: sub_68A670
// Address: 0x68a670
//
void __fastcall sub_68A670(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // edx

  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a2 + 32) = 0;
  v2 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 20) = v2;
  v3 = *(_DWORD *)(a1 + 40);
  if ( (v3 & 0x10) == 0 )
  {
    *(_DWORD *)(a1 + 40) = v3 | 0x10;
    sub_6E1DD0(a2);
    *(_QWORD *)(a2 + 8) = qword_4F06BC0;
    *(_QWORD *)(a2 + 20) = unk_4F061D8;
    qword_4F06BC0 = *(_QWORD *)(qword_4F04C68[0] + 488LL);
    *(_QWORD *)(a2 + 32) = sub_878CA0();
    sub_7296C0(a2 + 16);
  }
}
