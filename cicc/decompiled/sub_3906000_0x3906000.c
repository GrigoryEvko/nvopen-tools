// Function: sub_3906000
// Address: 0x3906000
//
__int64 __fastcall sub_3906000(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v8; // rdi
  __int64 v9; // r12
  void (__fastcall *v10)(__int64, __int64, __int64); // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // [rsp+8h] [rbp-98h]
  _QWORD v15[3]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v16; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v17; // [rsp+30h] [rbp-70h] BYREF
  __int16 v18; // [rsp+40h] [rbp-60h]
  _BYTE v19[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v20; // [rsp+60h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 8);
  v15[0] = a2;
  v15[1] = a3;
  v16 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL))(v8) + 8) == 9
    || (result = sub_3909510(*(_QWORD *)(a1 + 8), &v16), !(_BYTE)result) )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v14 = v16;
    v10 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v9 + 160LL);
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v18 = 261;
    v20 = 257;
    v17 = v15;
    v12 = sub_38C3B80(v11, (__int64)&v17, a4, a5, 0, (__int64)v19, -1, 0);
    v10(v9, v12, v14);
    return 0;
  }
  return result;
}
