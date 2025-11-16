// Function: sub_38FFC00
// Address: 0x38ffc00
//
__int64 __fastcall sub_38FFC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r12d
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v10; // rdi
  unsigned int v11; // [rsp+Ch] [rbp-44h] BYREF
  const char *v12; // [rsp+10h] [rbp-40h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v11 = 0;
  v5 = sub_38FF7C0(a1, (int *)&v11);
  if ( (_BYTE)v5 )
    return v5;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v8 + 904LL))(v8, v11, a4);
    return v5;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v14 = 1;
  v12 = "unexpected token in directive";
  v13 = 3;
  return (unsigned int)sub_3909CF0(v10, &v12, 0, 0, v6, v7);
}
