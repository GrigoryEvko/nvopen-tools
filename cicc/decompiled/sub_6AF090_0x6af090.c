// Function: sub_6AF090
// Address: 0x6af090
//
__int64 __fastcall sub_6AF090(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v11; // [rsp+10h] [rbp-310h]
  __int16 v12; // [rsp+1Ah] [rbp-306h]
  int v13; // [rsp+1Ch] [rbp-304h]
  __int64 v14; // [rsp+28h] [rbp-2F8h] BYREF
  _QWORD v15[44]; // [rsp+30h] [rbp-2F0h] BYREF
  _BYTE v16[68]; // [rsp+190h] [rbp-190h] BYREF
  __int64 v17; // [rsp+1D4h] [rbp-14Ch]
  __int64 v18; // [rsp+1DCh] [rbp-144h]

  if ( a1 )
  {
    v5 = *a1;
    v6 = sub_6E3DA0(*a1, 0);
    v7 = *(_QWORD *)(v5 + 64);
    v14 = *(_QWORD *)(v6 + 68);
    v13 = *(_DWORD *)(*a1 + 44LL);
    v12 = *(_WORD *)(*a1 + 48LL);
    sub_6F8800(v7, a1, v15);
    sub_6F69D0(v15, 0);
    if ( !(unsigned int)sub_8D2930(v15[0]) && !(unsigned int)sub_8D2E30(v15[0]) )
      sub_6E68E0(157, v15);
  }
  else
  {
    v14 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_6E1E00(4, v16, 0, 0);
    sub_69ED20((__int64)v15, 0, 0, 0);
    sub_6F69D0(v15, 0);
    v13 = qword_4F063F0;
    v12 = WORD2(qword_4F063F0);
    sub_6E2B30(v15, 0);
  }
  v11 = sub_72BA30(unk_4F06A51);
  v8 = sub_726700(23);
  *(_BYTE *)(v8 + 56) = 22;
  v9 = v8;
  *(_QWORD *)v8 = v11;
  *(_QWORD *)(v8 + 64) = sub_6F6F40(v15, 0);
  *(_QWORD *)(v9 + 28) = v14;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v9 + 24) )
  {
    sub_6E70E0(v9, v16);
    LODWORD(v17) = v14;
    WORD2(v17) = WORD2(v14);
    *(_QWORD *)dword_4F07508 = v17;
    LODWORD(v18) = v13;
    WORD2(v18) = v12;
    unk_4F061D8 = v18;
    sub_6E3280(v16, &dword_4F077C8);
    sub_6F6F40(v16, 0);
  }
  sub_6E70E0(v9, a2);
  sub_6F4D20(a2, 1, 1);
  if ( !a1 )
  {
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  *(_DWORD *)(a2 + 68) = v14;
  *(_WORD *)(a2 + 72) = WORD2(v14);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v13;
  *(_WORD *)(a2 + 80) = v12;
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  return sub_6E3280(a2, &v14);
}
