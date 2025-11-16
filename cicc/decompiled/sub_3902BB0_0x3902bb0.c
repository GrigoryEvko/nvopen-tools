// Function: sub_3902BB0
// Address: 0x3902bb0
//
__int64 __fastcall sub_3902BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v4; // r13
  __int64 v6; // r8
  __int64 v7; // r9
  const char *v8; // rax
  __int64 v9; // rdi
  const char *v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  _QWORD v14[2]; // [rsp+0h] [rbp-40h] BYREF
  char v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+11h] [rbp-2Fh]

  v4 = 0;
  if ( a3 == 5 )
  {
    if ( *(_DWORD *)a2 != 1836409902 || (v13 = 0, *(_BYTE *)(a2 + 4) != 112) )
      v13 = 1;
    v4 = v13 == 0;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v16 = 1;
    v11 = "expected string in '.dump' or '.load' directive";
    goto LABEL_8;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    v16 = 1;
    v11 = "unexpected token in '.dump' or '.load' directive";
LABEL_8:
    v12 = *(_QWORD *)(a1 + 8);
    v14[0] = v11;
    v15 = 3;
    return sub_3909CF0(v12, v14, 0, 0, v6, v7);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v16 = 1;
  v8 = "ignoring directive .dump for now";
  if ( !v4 )
    v8 = "ignoring directive .load for now";
  v9 = *(_QWORD *)(a1 + 8);
  v14[0] = v8;
  v15 = 3;
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, _QWORD, _QWORD))(*(_QWORD *)v9 + 120LL))(
           v9,
           a4,
           v14,
           0,
           0);
}
