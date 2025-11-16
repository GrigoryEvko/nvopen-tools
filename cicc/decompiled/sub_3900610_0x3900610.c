// Function: sub_3900610
// Address: 0x3900610
//
__int64 __fastcall sub_3900610(__int64 *a1)
{
  __int64 v1; // rax
  unsigned int v2; // eax
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v18; // [rsp+30h] [rbp-30h]

  v1 = *a1;
  v16[1] = 0;
  v16[0] = 0;
  v2 = (*(__int64 (__fastcall **)(_QWORD, _QWORD *))(**(_QWORD **)(v1 + 8) + 144LL))(*(_QWORD *)(v1 + 8), v16);
  if ( (_BYTE)v2 )
  {
    v14 = *a1;
    v17[0] = "expected identifier in directive";
    v18 = 259;
    return (unsigned int)sub_3909CF0(*(_QWORD *)(v14 + 8), v17, 0, 0, v3, v4);
  }
  else
  {
    v5 = v2;
    v6 = *a1;
    v15 = 0;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 8) + 40LL))(*(_QWORD *)(v6 + 8)) + 8) == 12
      || **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 40LL))(*(_QWORD *)(*a1 + 8)) + 8) == 13 )
    {
      v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 40LL))(*(_QWORD *)(*a1 + 8));
      v7 = sub_3909290(v13);
      LODWORD(result) = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(*a1 + 8) + 200LL))(
                          *(_QWORD *)(*a1 + 8),
                          &v15);
      if ( (_BYTE)result )
        return (unsigned int)result;
    }
    else
    {
      v7 = 0;
    }
    v8 = *a1;
    if ( (unsigned __int64)(v15 + 0x80000000LL) > 0xFFFFFFFF )
    {
      v18 = 259;
      v17[0] = "invalid '.rva' directive offset, can't be less than -2147483648 or greater than 2147483647";
      return (unsigned int)sub_3909790(*(_QWORD *)(v8 + 8), v7, v17, 0, 0);
    }
    else
    {
      v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v8 + 8) + 48LL))(*(_QWORD *)(v8 + 8));
      v17[0] = v16;
      v18 = 261;
      v10 = sub_38BF510(v9, (__int64)v17);
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 56LL))(*(_QWORD *)(*a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11 + 336LL))(v11, v10, v15);
      return v5;
    }
  }
}
