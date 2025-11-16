// Function: sub_EC9FA0
// Address: 0xec9fa0
//
__int64 __fastcall sub_EC9FA0(__int64 a1)
{
  _QWORD *v2; // rsi
  unsigned __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  const char *v12; // [rsp+10h] [rbp-50h] BYREF
  char v13; // [rsp+30h] [rbp-30h]
  char v14; // [rsp+31h] [rbp-2Fh]

  v2 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v3 = sub_E81A90(0, v2, 0, 0);
  v4 = *(_QWORD *)(a1 + 8);
  v11 = v3;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 40LL))(v4) + 8) == 9
    || (result = sub_ECD870(*(_QWORD *)(a1 + 8), &v11), !(_BYTE)result) )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v6 = v11;
      v7 = (_QWORD *)v5;
      v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      return sub_E9A6D0(v7, *(_QWORD *)(*(_QWORD *)(v8 + 288) + 8LL), v6);
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 8);
      v14 = 1;
      v12 = "expected end of directive";
      v13 = 3;
      return sub_ECE0E0(v10, &v12, 0, 0);
    }
  }
  return result;
}
