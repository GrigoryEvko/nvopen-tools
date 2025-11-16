// Function: sub_1F476F0
// Address: 0x1f476f0
//
void (*__fastcall sub_1F476F0(__int64 a1))()
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  void (*v12)(); // rax
  char v13; // si
  void (__fastcall *v14)(__int64, __int64); // r12
  __int64 v15; // rax
  void (*v16)(); // rax
  void *v17; // rsi
  void (*v18)(); // rax
  __int64 (*v19)(); // rax
  __int64 (__fastcall *v20)(__int64); // rax
  __int64 v21; // rdx
  void (*v22)(); // rax
  void (*result)(); // rax
  _QWORD *v24; // rsi
  _QWORD *v25; // rsi
  _QWORD *v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v31[6]; // [rsp+10h] [rbp-30h] BYREF

  v1 = qword_4FCC468;
  *(_BYTE *)(a1 + 202) = 1;
  if ( v1
    && (v1 != 18
     || *(_QWORD *)qword_4FCC460 ^ 0x752D6E6F6974706FLL | *(_QWORD *)(qword_4FCC460 + 8) ^ 0x696669636570736ELL
     || *(_WORD *)(qword_4FCC460 + 16) != 25701) )
  {
    v2 = sub_163A1D0();
    v6 = sub_163A430(v2, qword_4FCC460, qword_4FCC468, v3, v4, v5);
    v10 = sub_163A430(v2, (__int64)"machineinstr-printer", 20, v7, v8, v9);
    sub_1F45DE0(a1, *(_QWORD *)(v6 + 32), *(_QWORD *)(v10 + 32), 0, 1, 1);
  }
  v30[0] = (__int64)v31;
  sub_1F450A0(v30, "After Instruction Selection", (__int64)"");
  sub_1F46460(a1, (__int64)v30, v11);
  if ( (_QWORD *)v30[0] != v31 )
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  sub_1F46F00(a1, &unk_4FC350C, 1, 1, 1u);
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 288LL))(a1);
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 809LL) & 1) == 0 )
      goto LABEL_8;
  }
  else
  {
    sub_1F46F00(a1, &unk_4FC453C, 0, 1, 0);
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 809LL) & 1) == 0 )
      goto LABEL_8;
  }
  v24 = (_QWORD *)sub_1EECBB0();
  sub_1F46490(a1, v24, 1, 1, 0);
LABEL_8:
  v12 = *(void (**)())(*(_QWORD *)a1 + 304LL);
  if ( v12 != nullsub_764 )
    ((void (__fastcall *)(__int64))v12)(a1);
  if ( sub_1F475B0(a1) )
  {
    v13 = 1;
    v14 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 328LL);
  }
  else
  {
    if ( (__int64 (*)())qword_4FCB820 != sub_1F448E0 && (__int64 (*)())qword_4FCB820 != sub_1EB6E00 )
      sub_16BD130("Must use fast (default) register allocator for unoptimized regalloc.", 1u);
    v13 = 0;
    v14 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 320LL);
  }
  v15 = sub_1F475E0(a1, v13);
  v14(a1, v15);
  v16 = *(void (**)())(*(_QWORD *)a1 + 344LL);
  if ( v16 != nullsub_765 )
    ((void (__fastcall *)(__int64))v16)(a1);
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    sub_1F46F00(a1, &unk_4FC7F6C, 1, 1, 0);
    sub_1F46F00(a1, &unk_4FCA74C, 1, 1, 0);
  }
  if ( !(unsigned __int8)sub_1F46380(a1, &unk_4FC9148) )
  {
    v25 = (_QWORD *)sub_1EAE640();
    sub_1F46490(a1, v25, 1, 1, 0);
  }
  if ( (unsigned int)sub_1F45DD0(a1) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 352LL))(a1);
  v17 = &unk_4FC35EC;
  sub_1F46F00(a1, &unk_4FC35EC, 1, 1, 1u);
  v18 = *(void (**)())(*(_QWORD *)a1 + 360LL);
  if ( v18 != nullsub_766 )
    ((void (__fastcall *)(__int64))v18)(a1);
  if ( byte_4FCCD20 )
  {
    v17 = &unk_4FC40AC;
    sub_1F46F00(a1, &unk_4FC40AC, 1, 1, 0);
  }
  if ( (unsigned int)sub_1F45DD0(a1) )
  {
    v19 = *(__int64 (**)())(**(_QWORD **)(a1 + 208) + 72LL);
    if ( v19 == sub_16FF7A0 || !(unsigned __int8)v19() )
    {
      if ( LOBYTE(qword_4FCC200[20]) )
      {
        v17 = &unk_4FC786C;
        sub_1F46F00(a1, &unk_4FC786C, 1, 1, 0);
      }
      else
      {
        v17 = &unk_4FC8CB4;
        sub_1F46F00(a1, &unk_4FC8CB4, 1, 1, 0);
      }
    }
  }
  v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 368LL);
  if ( v20 == sub_1F47170 )
  {
    v17 = &unk_4FC3610;
    sub_1F46F00(a1, &unk_4FC3610, 0, 1, 0);
  }
  else if ( !(unsigned __int8)v20(a1) )
  {
    if ( (unsigned int)sub_1F45DD0(a1) )
      goto LABEL_42;
    goto LABEL_34;
  }
  if ( !byte_4FCCA80 )
  {
    if ( !(unsigned int)sub_1F45DD0(a1) )
      goto LABEL_34;
    goto LABEL_42;
  }
  v28 = sub_16BA580(a1, (__int64)v17, v21);
  v29 = (_QWORD *)sub_1D8E9B0(v28);
  sub_1F46490(a1, v29, 0, 0, 0);
  if ( (unsigned int)sub_1F45DD0(a1) )
LABEL_42:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 376LL))(a1);
LABEL_34:
  v22 = *(void (**)())(*(_QWORD *)a1 + 384LL);
  if ( v22 != nullsub_767 )
    ((void (__fastcall *)(__int64))v22)(a1);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 809LL) & 1) != 0 )
  {
    v27 = (_QWORD *)sub_1EEBCA0();
    sub_1F46490(a1, v27, 1, 1, 0);
  }
  sub_1F46F00(a1, &unk_4FC3604, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FCAACC, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC434C, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC35FC, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FCE6CC, 0, 1, 0);
  sub_1F46F00(a1, &unk_4FC84CC, 0, 1, 0);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 809LL) & 4) != 0
    && (unsigned int)sub_1F45DD0(a1)
    && dword_4FCC740 != 1
    && (!dword_4FCC740 || (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 809LL) & 8) != 0) )
  {
    v26 = (_QWORD *)sub_1E39EC0(dword_4FCC740 == 0);
    sub_1F46490(a1, v26, 1, 1, 0);
  }
  result = *(void (**)())(*(_QWORD *)a1 + 392LL);
  if ( result != nullsub_768 )
    result = (void (*)())((__int64 (__fastcall *)(__int64))result)(a1);
  *(_BYTE *)(a1 + 202) = 0;
  return result;
}
