// Function: sub_EC5E20
// Address: 0xec5e20
//
__int64 __fastcall sub_EC5E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v4; // r13
  int v6; // eax
  const char *v7; // rax
  __int64 v8; // rdi
  const char *v10; // rax
  __int64 v11; // rdi
  _QWORD v12[4]; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v4 = 0;
  if ( a3 == 5 )
  {
    if ( *(_DWORD *)a2 != 1836409902 || (v6 = 0, *(_BYTE *)(a2 + 4) != 112) )
      v6 = 1;
    v4 = v6 == 0;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v14 = 1;
    v10 = "expected string in '.dump' or '.load' directive";
    goto LABEL_11;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    v14 = 1;
    v10 = "unexpected token in '.dump' or '.load' directive";
LABEL_11:
    v11 = *(_QWORD *)(a1 + 8);
    v12[0] = v10;
    v13 = 3;
    return sub_ECE0E0(v11, v12, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v14 = 1;
  v7 = "ignoring directive .dump for now";
  if ( !v4 )
    v7 = "ignoring directive .load for now";
  v8 = *(_QWORD *)(a1 + 8);
  v12[0] = v7;
  v13 = 3;
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v8 + 168LL))(
           v8,
           a4,
           v12,
           0,
           0);
}
