// Function: sub_6DDBA0
// Address: 0x6ddba0
//
__int64 __fastcall sub_6DDBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // r15d
  __int64 j; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r9
  int v10; // eax
  __int64 i; // rdi
  __int64 v13; // [rsp+0h] [rbp-310h]
  __int16 v14; // [rsp+Eh] [rbp-302h]
  int v15; // [rsp+14h] [rbp-2FCh] BYREF
  __int64 v16; // [rsp+18h] [rbp-2F8h] BYREF
  _QWORD v17[44]; // [rsp+20h] [rbp-2F0h] BYREF
  _BYTE v18[68]; // [rsp+180h] [rbp-190h] BYREF
  __int64 v19; // [rsp+1C4h] [rbp-14Ch]
  __int64 v20; // [rsp+1CCh] [rbp-144h]

  v16 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  sub_69ED20((__int64)v17, 0, 0, 1);
  v5 = qword_4F063F0;
  v14 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( (unsigned int)sub_68B5C0(v17, &v15) )
  {
    if ( !v15 )
    {
      sub_6F69D0(v17, 0);
      v7 = sub_6F6F40(v17, 0);
      goto LABEL_11;
    }
  }
  else if ( !v15 )
  {
    sub_72C930(v17);
    sub_6E6260(a1);
    goto LABEL_8;
  }
  sub_6F40C0(v17);
  j = *(_QWORD *)&dword_4D03B80;
  v7 = sub_6F6F40(v17, 0);
  if ( !j )
  {
LABEL_11:
    for ( i = v17[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    for ( j = sub_8D46C0(i); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
  }
  v8 = sub_726700(23);
  *(_QWORD *)v8 = j;
  v9 = v8;
  *(_QWORD *)(v8 + 64) = v7;
  *(_BYTE *)(v8 + 56) = a2;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v8 + 24) )
  {
    v13 = v8;
    sub_6E70E0(v8, v18);
    LODWORD(v20) = v5;
    LODWORD(v19) = v16;
    WORD2(v19) = WORD2(v16);
    *(_QWORD *)dword_4F07508 = v19;
    WORD2(v20) = v14;
    *(_QWORD *)&dword_4F061D8 = v20;
    sub_6E3280(v18, &dword_4F077C8);
    sub_6F6F40(v18, 0);
    v9 = v13;
  }
  sub_6E70E0(v9, a1);
LABEL_8:
  v10 = v16;
  *(_DWORD *)(a1 + 76) = v5;
  *(_DWORD *)(a1 + 68) = v10;
  *(_WORD *)(a1 + 72) = WORD2(v16);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 68);
  *(_WORD *)(a1 + 80) = v14;
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(a1 + 76);
  return sub_6E3280(a1, &v16);
}
