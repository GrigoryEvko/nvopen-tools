// Function: sub_EC3310
// Address: 0xec3310
//
__int64 __fastcall sub_EC3310(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v12; // rdi
  const char *v13; // [rsp+0h] [rbp-60h] BYREF
  const char *v14; // [rsp+8h] [rbp-58h]
  const char *v15[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+30h] [rbp-30h]

  v6 = *(_QWORD *)(a1 + 8);
  v13 = 0;
  v14 = 0;
  v7 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v6 + 192LL))(v6, &v13);
  if ( (_BYTE)v7 )
    return v7;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v16 = 261;
    v15[0] = v13;
    v15[1] = v14;
    v9 = sub_E6C460(v8, v15);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v10 + 1056LL))(v10, v9, a4);
    return v7;
  }
  v12 = *(_QWORD *)(a1 + 8);
  v15[0] = "unexpected token in directive";
  v16 = 259;
  return (unsigned int)sub_ECE0E0(v12, v15, 0, 0);
}
