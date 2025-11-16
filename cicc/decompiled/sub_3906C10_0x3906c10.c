// Function: sub_3906C10
// Address: 0x3906c10
//
__int64 __fastcall sub_3906C10(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v17; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v15[0] = 0;
  v15[1] = 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v15) )
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v16[0] = v15;
    v17 = 261;
    v6 = sub_38BF510(v5, (__int64)v16);
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      v9 = sub_3909510(*(_QWORD *)(a1 + 8), &v14);
      if ( (_BYTE)v9 )
        return v9;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v10 + 344LL))(v10, v6, v14);
        return v9;
      }
    }
    v12 = *(_QWORD *)(a1 + 8);
    v16[0] = "unexpected token in directive";
    v17 = 259;
    return (unsigned int)sub_3909CF0(v12, v16, 0, 0, v7, v8);
  }
  v13 = *(_QWORD *)(a1 + 8);
  v16[0] = "expected identifier in directive";
  v17 = 259;
  return (unsigned int)sub_3909CF0(v13, v16, 0, 0, v3, v4);
}
