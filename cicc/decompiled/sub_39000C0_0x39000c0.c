// Function: sub_39000C0
// Address: 0x39000c0
//
__int64 __fastcall sub_39000C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // r12d
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v14; // rdi
  _QWORD v15[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  v6 = *(_QWORD *)(a1 + 8);
  v15[0] = 0;
  v15[1] = 0;
  v7 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v6 + 144LL))(v6, v15);
  if ( (_BYTE)v7 )
    return v7;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v16[0] = v15;
    v17 = 261;
    v11 = sub_38BF510(v10, (__int64)v16);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v12 + 872LL))(v12, v11, a4);
    return v7;
  }
  v14 = *(_QWORD *)(a1 + 8);
  v16[0] = "unexpected token in directive";
  v17 = 259;
  return (unsigned int)sub_3909CF0(v14, v16, 0, 0, v8, v9);
}
