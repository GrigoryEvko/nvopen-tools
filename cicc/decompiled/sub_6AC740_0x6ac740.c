// Function: sub_6AC740
// Address: 0x6ac740
//
__int64 __fastcall sub_6AC740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r13d
  __int64 v18; // rbx
  int v19; // eax
  int v20; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v21[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1;
  v7 = a2;
  v20 = 0;
  if ( a2 )
  {
    v21[0] = *(_QWORD *)(a2 + 68);
    v8 = qword_4D03C50;
  }
  else
  {
    v21[0] = *(_QWORD *)&dword_4F063F8;
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
      sub_6851C0(0x39u, v21);
    goto LABEL_6;
  }
  if ( (unsigned int)sub_6E9250(v21) )
LABEL_6:
    v20 = 1;
  v13 = sub_6AC060(1u, 0x3A2u, &v20);
  if ( v20 )
  {
    sub_6E6260(v6);
  }
  else
  {
    v15 = sub_72CBE0(1, 930, v9, v10, v11, v12);
    v16 = sub_73DBF0(113, v15, v13);
    sub_6E70E0(v16, v6);
  }
  result = sub_6E26D0(2, v6);
  if ( !v7 )
  {
    v17 = qword_4F063F0;
    v18 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    v19 = v21[0];
    *(_DWORD *)(v6 + 76) = v17;
    *(_DWORD *)(v6 + 68) = v19;
    LOWORD(v19) = WORD2(v21[0]);
    *(_WORD *)(v6 + 80) = v18;
    *(_WORD *)(v6 + 72) = v19;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
    unk_4F061D8 = *(_QWORD *)(v6 + 76);
    return sub_6E3280(v6, v21);
  }
  return result;
}
