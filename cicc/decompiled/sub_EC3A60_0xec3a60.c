// Function: sub_EC3A60
// Address: 0xec3a60
//
__int64 __fastcall sub_EC3A60(__int64 *a1)
{
  __int64 v1; // rax
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  const char *v14; // [rsp+10h] [rbp-60h] BYREF
  const char *v15; // [rsp+18h] [rbp-58h]
  const char *v16[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v17; // [rsp+40h] [rbp-30h]

  v1 = *a1;
  v15 = 0;
  v14 = 0;
  v2 = (*(__int64 (__fastcall **)(_QWORD, const char **))(**(_QWORD **)(v1 + 8) + 192LL))(*(_QWORD *)(v1 + 8), &v14);
  if ( (_BYTE)v2 )
  {
    v12 = *a1;
    v16[0] = "expected identifier in directive";
    v17 = 259;
    return (unsigned int)sub_ECE0E0(*(_QWORD *)(v12 + 8), v16, 0, 0);
  }
  else
  {
    v3 = v2;
    v4 = *a1;
    v13 = 0;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v4 + 8) + 40LL))(*(_QWORD *)(v4 + 8)) + 8) == 12
      || **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 40LL))(*(_QWORD *)(*a1 + 8)) + 8) == 13 )
    {
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 40LL))(*(_QWORD *)(*a1 + 8));
      v5 = sub_ECD690(v11);
      LODWORD(result) = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(*a1 + 8) + 256LL))(
                          *(_QWORD *)(*a1 + 8),
                          &v13);
      if ( (_BYTE)result )
        return (unsigned int)result;
    }
    else
    {
      v5 = 0;
    }
    v6 = *a1;
    if ( (unsigned __int64)(v13 + 0x80000000LL) > 0xFFFFFFFF )
    {
      v17 = 259;
      v16[0] = "invalid '.rva' directive offset, can't be less than -2147483648 or greater than 2147483647";
      return (unsigned int)sub_ECDA70(*(_QWORD *)(v6 + 8), v5, v16, 0, 0);
    }
    else
    {
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 8) + 48LL))(*(_QWORD *)(v6 + 8));
      v17 = 261;
      v16[0] = v14;
      v16[1] = v15;
      v8 = sub_E6C460(v7, v16);
      v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 8) + 56LL))(*(_QWORD *)(*a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v9 + 376LL))(v9, v8, v13);
      return v3;
    }
  }
}
