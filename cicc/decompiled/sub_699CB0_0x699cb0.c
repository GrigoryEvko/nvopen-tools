// Function: sub_699CB0
// Address: 0x699cb0
//
__int64 __fastcall sub_699CB0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v24; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v25[19]; // [rsp+10h] [rbp-220h] BYREF
  char v26; // [rsp+23h] [rbp-20Dh]
  _BYTE v27[68]; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v28; // [rsp+F4h] [rbp-13Ch]
  __int64 v29; // [rsp+FCh] [rbp-134h]

  sub_6E1DD0(&v24);
  v3 = v25;
  sub_6E1E00(5, v25, 0, 1);
  v26 |= 2u;
  v4 = sub_726700(37);
  *(_BYTE *)(v4 + 27) |= 2u;
  v5 = v4;
  v10 = sub_72CBE0(37, v25, v6, v7, v8, v9);
  *(_QWORD *)(v5 + 56) = a1;
  *(_QWORD *)v5 = v10;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v5 + 24) )
  {
    sub_6E70E0(v5, v27);
    WORD2(v28) = unk_4F077CC;
    LODWORD(v28) = dword_4F077C8;
    *(_QWORD *)dword_4F07508 = v28;
    v29 = *(_QWORD *)&dword_4F077C8;
    unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
    sub_6E3280(v27, &dword_4F077C8);
    v3 = 0;
    sub_6F6F40(v27, 0);
  }
  v11 = sub_726700(22);
  *(_BYTE *)(v11 + 27) |= 2u;
  v12 = v11;
  v17 = sub_72CBE0(22, v3, v13, v14, v15, v16);
  *(_QWORD *)(v12 + 56) = a2;
  *(_QWORD *)v12 = v17;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v12 + 24) )
  {
    sub_6E70E0(v12, v27);
    WORD2(v28) = unk_4F077CC;
    LODWORD(v28) = dword_4F077C8;
    *(_QWORD *)dword_4F07508 = v28;
    v29 = *(_QWORD *)&dword_4F077C8;
    unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
    sub_6E3280(v27, &dword_4F077C8);
    v3 = 0;
    sub_6F6F40(v27, 0);
  }
  *(_QWORD *)(v5 + 16) = v12;
  v18 = 23;
  v19 = sub_726700(23);
  *(_BYTE *)(v19 + 27) |= 2u;
  v20 = v19;
  v21 = sub_72C390();
  *(_BYTE *)(v20 + 56) = 78;
  *(_QWORD *)v20 = v21;
  v22 = qword_4D03C50;
  *(_QWORD *)(v20 + 64) = v5;
  if ( (*(_BYTE *)(v22 + 19) & 2) != 0 && *(_BYTE *)(v20 + 24) )
  {
    sub_6E70E0(v20, v27);
    WORD2(v28) = unk_4F077CC;
    LODWORD(v28) = dword_4F077C8;
    *(_QWORD *)dword_4F07508 = v28;
    v29 = *(_QWORD *)&dword_4F077C8;
    unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
    sub_6E3280(v27, &dword_4F077C8);
    v3 = 0;
    v18 = (__int64)v27;
    sub_6F6F40(v27, 0);
  }
  sub_6E2B30(v18, v3);
  sub_6E1DF0(v24);
  return v20;
}
