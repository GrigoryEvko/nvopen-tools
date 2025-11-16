// Function: sub_6AC910
// Address: 0x6ac910
//
__int64 __fastcall sub_6AC910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // r14d
  __int64 v20; // rbx
  int v21; // eax
  unsigned int v22; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v23[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1;
  v7 = a2;
  v22 = 0;
  if ( a2 )
  {
    v23[0] = *(_QWORD *)(a2 + 68);
    v8 = qword_4D03C50;
  }
  else
  {
    v23[0] = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a1, 0, a3, a4);
    a2 = 125;
    a1 = 27;
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    v8 = qword_4D03C50;
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  if ( (*(_BYTE *)(v8 + 19) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, a6) )
      sub_6851C0(0x39u, v23);
    goto LABEL_6;
  }
  if ( (unsigned int)sub_6E9250(v23) )
LABEL_6:
    v22 = 1;
  ++*(_BYTE *)(qword_4F061C8 + 75LL);
  v9 = sub_6AC060(0, 0x412u, &v22);
  ++*(_BYTE *)(qword_4F061C8 + 9LL);
  sub_7BE280(67, 253, 0, 0);
  v10 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 9LL);
  --*(_BYTE *)(v10 + 75);
  v11 = sub_6AC060(1u, 0x412u, &v22);
  v15 = v22;
  if ( v22 )
  {
    sub_6E6260(v6);
  }
  else
  {
    *(_QWORD *)(v9 + 16) = v11;
    v17 = sub_72CBE0(1, 1042, v15, v12, v13, v14);
    v18 = sub_73DBF0(114, v17, v9);
    sub_6E70E0(v18, v6);
  }
  result = sub_6E26D0(2, v6);
  if ( !v7 )
  {
    v19 = qword_4F063F0;
    v20 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    v21 = v23[0];
    *(_DWORD *)(v6 + 76) = v19;
    *(_DWORD *)(v6 + 68) = v21;
    LOWORD(v21) = WORD2(v23[0]);
    *(_WORD *)(v6 + 80) = v20;
    *(_WORD *)(v6 + 72) = v21;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
    unk_4F061D8 = *(_QWORD *)(v6 + 76);
    return sub_6E3280(v6, v23);
  }
  return result;
}
