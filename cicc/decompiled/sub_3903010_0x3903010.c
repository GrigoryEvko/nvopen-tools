// Function: sub_3903010
// Address: 0x3903010
//
__int64 __fastcall sub_3903010(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 result; // rax
  const char *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE v12[8]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v13[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v14[2]; // [rsp+20h] [rbp-30h] BYREF
  __int16 v15; // [rsp+30h] [rbp-20h]

  v2 = *(_QWORD *)(a1 + 8);
  v13[0] = 0;
  v13[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v13) )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v14[0] = "expected identifier in directive";
    v15 = 259;
    return sub_3909CF0(v11, v14, 0, 0, v3, v4);
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v14[0] = v13;
  v15 = 261;
  sub_38BF510(v5, (__int64)v14);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    goto LABEL_6;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  result = sub_3909510(*(_QWORD *)(a1 + 8), v12);
  if ( (_BYTE)result )
    return result;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    HIBYTE(v15) = 1;
    v9 = "directive '.lsym' is unsupported";
  }
  else
  {
LABEL_6:
    HIBYTE(v15) = 1;
    v9 = "unexpected token in '.lsym' directive";
  }
  v10 = *(_QWORD *)(a1 + 8);
  v14[0] = v9;
  LOBYTE(v15) = 3;
  return sub_3909CF0(v10, v14, 0, 0, v6, v7);
}
