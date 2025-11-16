// Function: sub_39012E0
// Address: 0x39012e0
//
__int64 __fastcall sub_39012E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r14d
  const char *v10; // rax
  __int64 v11; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int8 v16; // [rsp+1Eh] [rbp-62h] BYREF
  unsigned __int8 v17; // [rsp+1Fh] [rbp-61h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v20; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 8);
  v18[0] = 0;
  v18[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v6 + 144LL))(v6, v18) )
    return 1;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v16 = 0;
    v17 = 0;
    v9 = sub_38FFF90(a1, &v16, &v17);
    if ( (_BYTE)v9 )
      return 1;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      if ( (unsigned __int8)sub_38FFF90(a1, &v16, &v17) )
        return 1;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v19[0] = v18;
      v20 = 261;
      v14 = sub_38BF510(v13, (__int64)v19);
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v15 + 960LL))(
        v15,
        v14,
        v16,
        v17,
        a4);
      return v9;
    }
    HIBYTE(v20) = 1;
    v10 = "unexpected token in directive";
  }
  else
  {
    HIBYTE(v20) = 1;
    v10 = "you must specify one or both of @unwind or @except";
  }
  v11 = *(_QWORD *)(a1 + 8);
  v19[0] = v10;
  LOBYTE(v20) = 3;
  return (unsigned int)sub_3909CF0(v11, v19, 0, 0, v7, v8);
}
