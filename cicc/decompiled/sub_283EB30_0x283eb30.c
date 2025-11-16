// Function: sub_283EB30
// Address: 0x283eb30
//
_BYTE *__fastcall sub_283EB30(_BYTE *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v12; // rsi
  __int64 (__fastcall *v13)(__int64, __int64); // rax
  __int64 v14; // r8
  __int64 v15; // rdi
  _QWORD *v16; // rbx
  _QWORD *i; // r13
  __int64 v18; // rdx
  char *v19; // rsi
  _QWORD *v20; // rdi
  void (__fastcall *v21)(_QWORD *, char *, __int64, char *); // r9
  char *(*v22)(); // rax
  __int64 v23; // rax
  bool v24; // zf
  __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+8h] [rbp-98h]
  char v29[8]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v30; // [rsp+18h] [rbp-88h]
  char v31; // [rsp+2Ch] [rbp-74h]
  char v32[16]; // [rsp+30h] [rbp-70h] BYREF
  char v33[8]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v34; // [rsp+48h] [rbp-58h]
  char v35; // [rsp+5Ch] [rbp-44h]
  char v36[64]; // [rsp+60h] [rbp-40h] BYREF

  if ( (unsigned __int8)sub_283D940(a7, *a3, (__int64)a2) )
  {
    v12 = *a3;
    v13 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a3 + 16LL);
    if ( v13 == sub_2302D00 )
      sub_283EA00((__int64)v29, v12 + 8, a2, a4, a5, a6);
    else
      ((void (__fastcall *)(char *, __int64, __int64 *, __int64, __int64, __int64))v13)(v29, v12, a2, a4, a5, a6);
    v14 = *a3;
    v15 = *a7;
    if ( *(_BYTE *)(a6 + 24) )
    {
      if ( v15 )
      {
        v16 = *(_QWORD **)(v15 + 576);
        for ( i = &v16[4 * *(unsigned int *)(v15 + 584)]; i != v16; v14 = v27 )
        {
          v22 = *(char *(**)())(*(_QWORD *)v14 + 32LL);
          if ( v22 == sub_230FAA0 )
          {
            v18 = 149;
            v19 = "PassManager<llvm::Loop, llvm::AnalysisManager<llvm::Loop, llvm::LoopStandardAnalysisResults&>, llvm::LoopStandardAnalysisResults&, llvm::LPMUpdater&>]";
          }
          else
          {
            v28 = v14;
            v23 = ((__int64 (__fastcall *)(__int64))v22)(v14);
            v14 = v28;
            v19 = (char *)v23;
          }
          v20 = v16;
          v21 = *(void (__fastcall **)(_QWORD *, char *, __int64, char *))(v16[3] & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v16[3] & 2) == 0 )
            v20 = (_QWORD *)*v16;
          v16 += 4;
          v27 = v14;
          v21(v20, v19, v18, v29);
        }
      }
    }
    else
    {
      sub_283D2B0(v15, *a3, (__int64)a2, (__int64)v29);
    }
    sub_C8CF70((__int64)a1, a1 + 32, 2, (__int64)v32, (__int64)v29);
    sub_C8CF70((__int64)(a1 + 48), a1 + 80, 2, (__int64)v36, (__int64)v33);
    v24 = v35 == 0;
    a1[96] = 1;
    if ( v24 )
      _libc_free(v34);
    if ( !v31 )
      _libc_free(v30);
  }
  else
  {
    a1[96] = 0;
  }
  return a1;
}
