// Function: sub_39001B0
// Address: 0x39001b0
//
__int64 __fastcall sub_39001B0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  const char *v11; // rax
  __int64 v12; // rdi
  _QWORD v13[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v15; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v13[0] = 0;
  v13[1] = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v13);
  if ( (_BYTE)v3 )
  {
    HIBYTE(v15) = 1;
    v11 = "expected identifier in directive";
  }
  else
  {
    v6 = v3;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v14[0] = v13;
      v15 = 261;
      v8 = sub_38BF510(v7, (__int64)v14);
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 312LL))(v9, v8);
      return v6;
    }
    HIBYTE(v15) = 1;
    v11 = "unexpected token in directive";
  }
  v12 = *(_QWORD *)(a1 + 8);
  v14[0] = v11;
  LOBYTE(v15) = 3;
  return (unsigned int)sub_3909CF0(v12, v14, 0, 0, v4, v5);
}
