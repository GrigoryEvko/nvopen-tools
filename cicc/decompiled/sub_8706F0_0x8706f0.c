// Function: sub_8706F0
// Address: 0x8706f0
//
void __fastcall sub_8706F0(_BYTE *a1, int a2)
{
  _BYTE *v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int *v14; // rsi
  unsigned int v15[14]; // [rsp+18h] [rbp-38h] BYREF

  if ( a1 )
  {
    v3 = a1;
  }
  else
  {
    sub_86F7D0(0x6Fu, dword_4F07508);
    v14 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
    if ( !v14 )
      v14 = &dword_4F063F8;
    v3 = sub_86E480(0x13u, v14);
    if ( !dword_4F04C3C )
      sub_8699D0((__int64)v3, 21, 0);
    sub_86E330((__int64)v3);
  }
  v4 = *((_QWORD *)v3 + 9);
  *(_QWORD *)(v4 + 8) = sub_86FD00(0, a2, 0, 0, 0, 0);
  *((_QWORD *)v3 + 1) = *(_QWORD *)&dword_4F061D8;
  *(_QWORD *)v15 = *(_QWORD *)&dword_4F063F8;
  if ( (unsigned int)sub_7BE280(0x96u, 530, 0, 0, v5, v6) )
  {
    do
    {
      v7 = qword_4D03B98 + 176LL * unk_4D03B90;
      *(_QWORD *)(v7 + 116) |= qword_4F5FD78;
      v8 = dword_4F5FD80 | *(_DWORD *)(v7 + 124);
      *(_QWORD *)(v7 + 48) = 0;
      *(_DWORD *)(v7 + 124) = v8;
      v9 = *(_QWORD *)(v7 + 104);
      *(_QWORD *)(v7 + 56) = 0;
      LODWORD(v7) = *(_DWORD *)(v7 + 112);
      qword_4F5FD78 = v9;
      dword_4F5FD80 = v7;
      sub_65D790((__int64)v3, v15, a1 != 0);
      *(_QWORD *)v15 = *(_QWORD *)&dword_4F063F8;
    }
    while ( (unsigned int)sub_7BE800(0x96u, v15, v10, v11, v12, v13) );
  }
  if ( dword_4F077C4 == 2 )
    sub_733F40();
  sub_86F030();
}
