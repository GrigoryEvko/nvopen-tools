// Function: sub_283D940
// Address: 0x283d940
//
__int64 __fastcall sub_283D940(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 (*v6)(); // rdx
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  char *v9; // rsi
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rdi
  char *(*v13)(); // rax
  bool v15; // zf
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  char *v18; // rsi
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rdi
  char *(*v22)(); // rax
  _QWORD *v23; // rbx
  __int64 v24; // rdx
  char *v25; // rsi
  _QWORD *v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdi
  char *(*v29)(); // rax
  __int64 v31; // [rsp+10h] [rbp-50h]
  _QWORD *j; // [rsp+10h] [rbp-50h]
  _QWORD *i; // [rsp+18h] [rbp-48h]
  unsigned __int8 v34; // [rsp+18h] [rbp-48h]
  _QWORD v35[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *a1;
  if ( !*a1 )
    return 1;
  v6 = *(__int64 (**)())(*(_QWORD *)a2 + 40LL);
  if ( v6 == sub_22FC9A0 )
    goto LABEL_3;
  v15 = ((unsigned __int8 (__fastcall *)(__int64))v6)(a2) == 0;
  v3 = *a1;
  if ( !v15 )
    goto LABEL_3;
  v16 = *(_QWORD **)v3;
  v31 = *(_QWORD *)v3 + 32LL * *(unsigned int *)(v3 + 8);
  if ( v31 == *(_QWORD *)v3 )
    goto LABEL_3;
  v34 = 1;
  do
  {
    v35[0] = 0;
    v20 = (_QWORD *)sub_22077B0(0x10u);
    if ( v20 )
    {
      v20[1] = a3;
      *v20 = &unk_4A09EA8;
    }
    v21 = v35[0];
    v35[0] = v20;
    if ( v21 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
    v22 = *(char *(**)())(*(_QWORD *)a2 + 32LL);
    if ( v22 == sub_230FAA0 )
    {
      v17 = 149;
      v18 = "PassManager<llvm::Loop, llvm::AnalysisManager<llvm::Loop, llvm::LoopStandardAnalysisResults&>, llvm::LoopStandardAnalysisResults&, llvm::LPMUpdater&>]";
    }
    else
    {
      v18 = (char *)((__int64 (__fastcall *)(__int64))v22)(a2);
    }
    v19 = v16;
    if ( (v16[3] & 2) == 0 )
      v19 = (_QWORD *)*v16;
    v34 &= (*(__int64 (__fastcall **)(_QWORD *, char *, __int64, _QWORD *))(v16[3] & 0xFFFFFFFFFFFFFFF8LL))(
             v19,
             v18,
             v17,
             v35);
    if ( v35[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v35[0] + 8LL))(v35[0]);
    v16 += 4;
  }
  while ( (_QWORD *)v31 != v16 );
  v3 = *a1;
  if ( v34 )
  {
LABEL_3:
    v7 = *(_QWORD **)(v3 + 288);
    for ( i = &v7[4 * *(unsigned int *)(v3 + 296)]; i != v7; v7 += 4 )
    {
      v35[0] = 0;
      v11 = (_QWORD *)sub_22077B0(0x10u);
      if ( v11 )
      {
        v11[1] = a3;
        *v11 = &unk_4A09EA8;
      }
      v12 = v35[0];
      v35[0] = v11;
      if ( v12 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
      v13 = *(char *(**)())(*(_QWORD *)a2 + 32LL);
      if ( v13 == sub_230FAA0 )
      {
        v8 = 149;
        v9 = "PassManager<llvm::Loop, llvm::AnalysisManager<llvm::Loop, llvm::LoopStandardAnalysisResults&>, llvm::LoopStandardAnalysisResults&, llvm::LPMUpdater&>]";
      }
      else
      {
        v9 = (char *)((__int64 (__fastcall *)(__int64))v13)(a2);
      }
      v10 = v7;
      if ( (v7[3] & 2) == 0 )
        v10 = (_QWORD *)*v7;
      (*(void (__fastcall **)(_QWORD *, char *, __int64, _QWORD *))(v7[3] & 0xFFFFFFFFFFFFFFF8LL))(v10, v9, v8, v35);
      if ( v35[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v35[0] + 8LL))(v35[0]);
    }
    return 1;
  }
  v23 = *(_QWORD **)(v3 + 144);
  for ( j = &v23[4 * *(unsigned int *)(v3 + 152)]; j != v23; v23 += 4 )
  {
    v35[0] = 0;
    v27 = (_QWORD *)sub_22077B0(0x10u);
    if ( v27 )
    {
      v27[1] = a3;
      *v27 = &unk_4A09EA8;
    }
    v28 = v35[0];
    v35[0] = v27;
    if ( v28 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
    v29 = *(char *(**)())(*(_QWORD *)a2 + 32LL);
    if ( v29 == sub_230FAA0 )
    {
      v24 = 149;
      v25 = "PassManager<llvm::Loop, llvm::AnalysisManager<llvm::Loop, llvm::LoopStandardAnalysisResults&>, llvm::LoopStandardAnalysisResults&, llvm::LPMUpdater&>]";
    }
    else
    {
      v25 = (char *)((__int64 (__fastcall *)(__int64))v29)(a2);
    }
    v26 = v23;
    if ( (v23[3] & 2) == 0 )
      v26 = (_QWORD *)*v23;
    (*(void (__fastcall **)(_QWORD *, char *, __int64, _QWORD *))(v23[3] & 0xFFFFFFFFFFFFFFF8LL))(v26, v25, v24, v35);
    if ( v35[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v35[0] + 8LL))(v35[0]);
  }
  return v34;
}
