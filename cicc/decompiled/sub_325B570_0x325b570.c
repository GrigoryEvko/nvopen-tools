// Function: sub_325B570
// Address: 0x325b570
//
__int64 __fastcall sub_325B570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 (*v10)(); // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 (__fastcall *v13)(__int64 *, __int64, __int64); // rax
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // r15
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 (__fastcall *v19)(__int64, __int64, unsigned __int64); // rbx
  unsigned __int64 v20; // rax
  unsigned int v22; // [rsp+8h] [rbp-68h]
  __int64 v23[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  v9 = *(unsigned int *)(a2 + 744);
  v22 = *(_DWORD *)(a2 + 744);
  if ( (_DWORD)v9 == 0x7FFFFFFF )
  {
    v15 = 0;
  }
  else
  {
    v10 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 16LL) + 136LL);
    if ( v10 == sub_2DD19D0 )
      BUG();
    v11 = v10();
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
    v13 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64))(*(_QWORD *)v11 + 240LL);
    if ( v13 == sub_2FDBC60 )
    {
      LODWORD(v23[0]) = 0;
      v14 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64 *))(*(_QWORD *)v11 + 224LL))(
              v11,
              v12,
              v22,
              v23);
    }
    else
    {
      v14 = v13((__int64 *)v11, v12, v22);
    }
    v15 = v14;
  }
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL);
  v23[0] = a3;
  v23[1] = a4;
  v24 = 261;
  v17 = sub_E6C770((__int64)v16, v23, v9, a4, (const char *)v23, a6);
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v18 + 272LL);
  v20 = sub_E81A90(v15, v16, 0, 0);
  return v19(v18, v17, v20);
}
