// Function: sub_283D2B0
// Address: 0x283d2b0
//
void __fastcall sub_283D2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rbx
  __int64 v8; // rdx
  char *v9; // rsi
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  char *(*v13)(); // rax
  _QWORD *i; // [rsp-50h] [rbp-50h]
  _QWORD *v15; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 )
  {
    v4 = *(_QWORD **)(a1 + 432);
    for ( i = &v4[4 * *(unsigned int *)(a1 + 440)]; i != v4; v4 += 4 )
    {
      v15 = 0;
      v11 = (_QWORD *)sub_22077B0(0x10u);
      if ( v11 )
      {
        v11[1] = a3;
        *v11 = &unk_4A09EA8;
      }
      v12 = v15;
      v15 = v11;
      if ( v12 )
        (*(void (__fastcall **)(_QWORD *))(*v12 + 8LL))(v12);
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
      v10 = v4;
      if ( (v4[3] & 2) == 0 )
        v10 = (_QWORD *)*v4;
      (*(void (__fastcall **)(_QWORD *, char *, __int64, _QWORD **, __int64))(v4[3] & 0xFFFFFFFFFFFFFFF8LL))(
        v10,
        v9,
        v8,
        &v15,
        a4);
      if ( v15 )
        (*(void (__fastcall **)(_QWORD *))(*v15 + 8LL))(v15);
    }
  }
}
