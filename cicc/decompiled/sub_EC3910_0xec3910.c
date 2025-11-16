// Function: sub_EC3910
// Address: 0xec3910
//
__int64 __fastcall sub_EC3910(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h]
  _QWORD v13[4]; // [rsp+10h] [rbp-50h] BYREF
  char v14; // [rsp+30h] [rbp-30h]
  char v15; // [rsp+31h] [rbp-2Fh]

  v5 = *(_QWORD *)(a1 + 8);
  v11 = 0;
  v12 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5) + 8) != 46
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 37 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v15 = 1;
    v13[0] = "a handler attribute must begin with '@' or '%'";
    v14 = 3;
    return sub_ECE0E0(v10, v13, 0, 0);
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v7 = sub_ECD690(v6);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  result = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &v11);
  if ( (_BYTE)result || v12 != 6 )
    goto LABEL_7;
  if ( *(_DWORD *)v11 == 1769434741 && *(_WORD *)(v11 + 4) == 25710 )
  {
    *a2 = 1;
    return result;
  }
  if ( *(_DWORD *)v11 != 1701017701 || *(_WORD *)(v11 + 4) != 29808 )
  {
LABEL_7:
    v9 = *(_QWORD *)(a1 + 8);
    v15 = 1;
    v13[0] = "expected @unwind or @except";
    v14 = 3;
    return sub_ECDA70(v9, v7, v13, 0, 0);
  }
  *a3 = 1;
  return result;
}
