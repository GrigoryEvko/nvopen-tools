// Function: sub_EC6290
// Address: 0xec6290
//
__int64 __fastcall sub_EC6290(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 result; // rax
  const char *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi
  _BYTE v8[8]; // [rsp+8h] [rbp-58h] BYREF
  const char *v9; // [rsp+10h] [rbp-50h] BYREF
  const char *v10; // [rsp+18h] [rbp-48h]
  const char *v11[4]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v12; // [rsp+40h] [rbp-20h]

  v2 = *(_QWORD *)(a1 + 8);
  v9 = 0;
  v10 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v9) )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v11[0] = "expected identifier in directive";
    v12 = 259;
    return sub_ECE0E0(v7, v11, 0, 0);
  }
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v12 = 261;
  v11[0] = v9;
  v11[1] = v10;
  sub_E6C460(v3, v11);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
    goto LABEL_6;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  result = sub_ECD870(*(_QWORD *)(a1 + 8), v8);
  if ( (_BYTE)result )
    return result;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    HIBYTE(v12) = 1;
    v5 = "directive '.lsym' is unsupported";
  }
  else
  {
LABEL_6:
    HIBYTE(v12) = 1;
    v5 = "unexpected token in '.lsym' directive";
  }
  v6 = *(_QWORD *)(a1 + 8);
  v11[0] = v5;
  LOBYTE(v12) = 3;
  return sub_ECE0E0(v6, v11, 0, 0);
}
