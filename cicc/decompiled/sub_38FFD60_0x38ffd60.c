// Function: sub_38FFD60
// Address: 0x38ffd60
//
__int64 __fastcall sub_38FFD60(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v6; // rdi
  unsigned int v7; // [rsp+8h] [rbp-38h] BYREF
  const char *v8; // [rsp+10h] [rbp-30h] BYREF
  char v9; // [rsp+20h] [rbp-20h]
  char v10; // [rsp+21h] [rbp-1Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *))(**(_QWORD **)(a1 + 8) + 200LL))(*(_QWORD *)(a1 + 8), &v7);
  if ( (_BYTE)v1 )
    return v1;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 280LL))(v4, v7);
    return v1;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v10 = 1;
  v8 = "unexpected token in directive";
  v9 = 3;
  return (unsigned int)sub_3909CF0(v6, &v8, 0, 0, v2, v3);
}
