// Function: sub_31F8650
// Address: 0x31f8650
//
__int64 __fastcall sub_31F8650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 *v17; // rdi
  __int64 v18; // rax
  void (*v19)(); // rax
  const char *v21; // [rsp+0h] [rbp-50h] BYREF
  char v22; // [rsp+20h] [rbp-30h]
  char v23; // [rsp+21h] [rbp-2Fh]

  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(v6 + 2480);
  v8 = v6 + 8;
  if ( !v7 )
    v7 = v8;
  v12 = sub_E6C430(v7, a2, a3, a4, a5);
  v13 = *(_QWORD *)(a1 + 16);
  v14 = *(_QWORD *)(v13 + 2480);
  v15 = v13 + 8;
  if ( !v14 )
    v14 = v15;
  v16 = sub_E6C430(v14, a2, v9, v10, v11);
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 528) + 536LL))(
    *(_QWORD *)(a1 + 528),
    (unsigned int)a2,
    4);
  v17 = *(__int64 **)(a1 + 528);
  v18 = *v17;
  v23 = 1;
  v21 = "Subsection size";
  v19 = *(void (**)())(v18 + 120);
  v22 = 3;
  if ( v19 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, const char **, __int64))v19)(v17, &v21, 1);
    v17 = *(__int64 **)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64))(*v17 + 832))(v17, v16, v12, 4);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 528) + 208LL))(*(_QWORD *)(a1 + 528), v12, 0);
  return v16;
}
