// Function: sub_38FFF90
// Address: 0x38fff90
//
__int64 __fastcall sub_38FFF90(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 result; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  __int64 v14; // [rsp+8h] [rbp-48h]
  _QWORD v15[2]; // [rsp+10h] [rbp-40h] BYREF
  char v16; // [rsp+20h] [rbp-30h]
  char v17; // [rsp+21h] [rbp-2Fh]

  v5 = *(_QWORD *)(a1 + 8);
  v13 = 0;
  v14 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5) + 8) == 45 )
  {
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v9 = sub_3909290(v8);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    result = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), &v13);
    if ( (_BYTE)result || v14 != 6 )
      goto LABEL_4;
    if ( *(_DWORD *)v13 == 1769434741 && *(_WORD *)(v13 + 4) == 25710 )
    {
      *a2 = 1;
    }
    else
    {
      if ( *(_DWORD *)v13 != 1701017701 || *(_WORD *)(v13 + 4) != 29808 )
      {
LABEL_4:
        v11 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v15[0] = "expected @unwind or @except";
        v16 = 3;
        return sub_3909790(v11, v9, v15, 0, 0);
      }
      *a3 = 1;
    }
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 8);
    v17 = 1;
    v15[0] = "a handler attribute must begin with '@'";
    v16 = 3;
    return sub_3909CF0(v12, v15, 0, 0, v6, v7);
  }
  return result;
}
