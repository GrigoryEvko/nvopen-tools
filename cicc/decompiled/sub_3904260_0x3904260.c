// Function: sub_3904260
// Address: 0x3904260
//
__int64 __fastcall sub_3904260(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  const char *v11; // rax
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  _DWORD *v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  const char *v19; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+10h] [rbp-30h]
  char v21; // [rsp+11h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 4 )
  {
    v21 = 1;
    v11 = "invalid OS major version number, integer expected";
    goto LABEL_9;
  }
  v8 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
  v10 = *(_QWORD **)(v8 + 24);
  if ( !v9 )
    v10 = (_QWORD *)*v10;
  if ( (unsigned __int64)v10 - 1 > 0xFFFE )
  {
    v21 = 1;
    v11 = "invalid OS major version number";
    goto LABEL_9;
  }
  *a2 = (_DWORD)v10;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    v21 = 1;
    v11 = "OS minor version number required, comma expected";
    goto LABEL_9;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 4 )
  {
    v21 = 1;
    v11 = "invalid OS minor version number, integer expected";
LABEL_9:
    v12 = *(_QWORD *)(a1 + 8);
    v19 = v11;
    v20 = 3;
    return sub_3909CF0(v12, &v19, 0, 0, v6, v7);
  }
  v14 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  v9 = *(_DWORD *)(v14 + 32) <= 0x40u;
  v15 = *(_QWORD **)(v14 + 24);
  if ( !v9 )
    v15 = (_QWORD *)*v15;
  if ( (unsigned __int64)v15 > 0xFF )
  {
    v21 = 1;
    v11 = "invalid OS minor version number";
    goto LABEL_9;
  }
  *a3 = (_DWORD)v15;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  *a4 = 0;
  v16 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  result = 0;
  if ( *v16 != 9 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 4 )
      {
        v17 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
        v9 = *(_DWORD *)(v17 + 32) <= 0x40u;
        v18 = *(_QWORD **)(v17 + 24);
        if ( !v9 )
          v18 = (_QWORD *)*v18;
        if ( (unsigned __int64)v18 <= 0xFF )
        {
          *a4 = (_DWORD)v18;
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
          return 0;
        }
        v21 = 1;
        v11 = "invalid OS update version number";
      }
      else
      {
        v21 = 1;
        v11 = "invalid OS update version number, integer expected";
      }
    }
    else
    {
      v21 = 1;
      v11 = "invalid OS update specifier, comma expected";
    }
    goto LABEL_9;
  }
  return result;
}
