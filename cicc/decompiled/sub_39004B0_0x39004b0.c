// Function: sub_39004B0
// Address: 0x39004b0
//
__int64 __fastcall sub_39004B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // r13
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  _DWORD *v16; // [rsp+0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+8h] [rbp-48h]
  _QWORD v18[2]; // [rsp+10h] [rbp-40h] BYREF
  char v19; // [rsp+20h] [rbp-30h]
  char v20; // [rsp+21h] [rbp-2Fh]

  v4 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v17 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7) + 8) == 45 )
  {
    v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v14 = sub_3909290(v13);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _DWORD **))(**(_QWORD **)(a1 + 8) + 144LL))(
            *(_QWORD *)(a1 + 8),
            &v16) )
    {
      if ( v17 != 4 || *v16 != 1701080931 )
      {
        v15 = *(_QWORD *)(a1 + 8);
        v20 = 1;
        v18[0] = "expected @code";
        v19 = 3;
        return sub_3909790(v15, v14, v18, 0, 0);
      }
      v4 = 1;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v10 + 944LL))(v10, v4, a4);
    return 0;
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 8);
    v20 = 1;
    v18[0] = "unexpected token in directive";
    v19 = 3;
    return sub_3909CF0(v12, v18, 0, 0, v8, v9);
  }
}
