// Function: sub_ECEEB0
// Address: 0xeceeb0
//
__int64 __fastcall sub_ECEEB0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rax
  const char *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  const char *v18; // [rsp+0h] [rbp-50h] BYREF
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+18h] [rbp-38h]
  __int16 v22; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( *(_DWORD *)v2 != 2 )
  {
    v14 = *(_QWORD *)(v2 + 8);
    v15 = *(_QWORD *)(v2 + 16);
    v16 = *(_QWORD *)(a1 + 24);
    v18 = "Expected label after .type directive, got: ";
    v20 = v14;
    v22 = 1285;
    v19 = 43;
    v21 = v15;
    v17 = sub_ECD6A0(v2);
    return sub_ECDA70(v16, v17, (__int64)&v18, 0, 0);
  }
  v3 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8)) + 8);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  v5 = *(const char **)(v4 + 8);
  v6 = *(_QWORD *)(v4 + 16);
  v22 = 261;
  v18 = v5;
  v19 = v6;
  v7 = sub_E6C460(v3, &v18);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( *(_DWORD *)v8 != 26
    || ((*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8)),
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
        *(_DWORD *)v8 != 46)
    || ((*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8)),
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
        *(_DWORD *)v8 != 2) )
  {
    v9 = *(_QWORD *)(v8 + 8);
    v10 = *(_QWORD *)(v8 + 16);
    v19 = 39;
    v18 = "Expected label,@type declaration, got: ";
    v11 = *(_QWORD *)(a1 + 24);
LABEL_4:
    v21 = v10;
    v20 = v9;
    v22 = 1285;
    v12 = sub_ECD6A0(v8);
    return sub_ECDA70(v11, v12, (__int64)&v18, 0, 0);
  }
  v10 = *(_QWORD *)(v8 + 16);
  v9 = *(_QWORD *)(v8 + 8);
  if ( v10 == 8 )
  {
    if ( *(_QWORD *)v9 == 0x6E6F6974636E7566LL )
    {
      *(_DWORD *)(v7 + 32) = 0;
      *(_BYTE *)(v7 + 36) = 1;
      if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8))
                                             + 288)
                                 + 8LL)
                     + 152LL) )
        *(_BYTE *)(v7 + 42) = 1;
      goto LABEL_11;
    }
    goto LABEL_15;
  }
  if ( v10 != 6 )
  {
LABEL_15:
    v19 = 26;
    v11 = *(_QWORD *)(a1 + 24);
    v18 = "Unknown WASM symbol type: ";
    goto LABEL_4;
  }
  if ( *(_DWORD *)v9 == 1651469415 && *(_WORD *)(v9 + 4) == 27745 )
  {
    *(_DWORD *)(v7 + 32) = 2;
    *(_BYTE *)(v7 + 36) = 1;
  }
  else
  {
    if ( *(_DWORD *)v9 != 1701470831 || *(_WORD *)(v9 + 4) != 29795 )
      goto LABEL_15;
    *(_DWORD *)(v7 + 32) = 1;
    *(_BYTE *)(v7 + 36) = 1;
  }
LABEL_11:
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)(*(_QWORD *)(a1 + 32) + 8LL) != 9 )
    return sub_ECEAE0(a1, "EOL");
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  return 0;
}
