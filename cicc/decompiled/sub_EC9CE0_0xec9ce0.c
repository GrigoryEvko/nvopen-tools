// Function: sub_EC9CE0
// Address: 0xec9ce0
//
__int64 __fastcall sub_EC9CE0(__int64 a1, size_t a2, size_t a3, int a4, unsigned int a5)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 result; // rax
  _QWORD *v13; // [rsp+0h] [rbp-B0h]
  __int64 v14; // [rsp+8h] [rbp-A8h]
  __int64 v15; // [rsp+18h] [rbp-98h] BYREF
  size_t v16[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v17; // [rsp+40h] [rbp-70h]
  _BYTE v18[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v19; // [rsp+70h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 8);
  v15 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 40LL))(v9) + 8) == 9
    || (result = sub_ECD870(*(_QWORD *)(a1 + 8), &v15), !(_BYTE)result) )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v13 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v14 = v15;
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v19 = 257;
    v16[0] = a2;
    v16[1] = a3;
    v17 = 261;
    v11 = sub_E71CB0(v10, v16, a4, a5, 0, (__int64)v18, 0, -1, 0);
    sub_E9A6D0(v13, v11, v14);
    return 0;
  }
  return result;
}
