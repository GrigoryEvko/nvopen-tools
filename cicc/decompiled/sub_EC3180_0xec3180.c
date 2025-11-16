// Function: sub_EC3180
// Address: 0xec3180
//
__int64 __fastcall sub_EC3180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v8; // rdi
  unsigned int v9; // [rsp+8h] [rbp-58h] BYREF
  const char *v10; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+30h] [rbp-30h]
  char v12; // [rsp+31h] [rbp-2Fh]

  v5 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *))(**(_QWORD **)(a1 + 8) + 256LL))(*(_QWORD *)(a1 + 8), &v9);
  if ( (_BYTE)v5 )
    return v5;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v6 + 1112LL))(v6, v9, a4);
    return v5;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v10 = "unexpected token in directive";
  v11 = 3;
  return (unsigned int)sub_ECE0E0(v8, &v10, 0, 0);
}
