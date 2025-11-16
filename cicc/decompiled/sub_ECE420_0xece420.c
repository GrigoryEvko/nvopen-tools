// Function: sub_ECE420
// Address: 0xece420
//
__int64 __fastcall sub_ECE420(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdi
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  _QWORD *v13; // rax
  unsigned __int64 v14; // r14
  _QWORD *v15; // rax
  unsigned __int64 v16; // rax
  const char *v17; // rax
  __int64 v18; // rdi
  __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-B0h]
  __int64 v22; // [rsp+8h] [rbp-A8h]
  void (__fastcall *v23)(__int64, unsigned __int64, unsigned __int64, __int64); // [rsp+10h] [rbp-A0h]
  __int64 v24; // [rsp+18h] [rbp-98h]
  __int64 v25; // [rsp+28h] [rbp-88h] BYREF
  const char *v26; // [rsp+30h] [rbp-80h] BYREF
  __int64 v27; // [rsp+38h] [rbp-78h]
  const char *v28; // [rsp+40h] [rbp-70h] BYREF
  __int64 v29; // [rsp+48h] [rbp-68h]
  const char *v30; // [rsp+50h] [rbp-60h] BYREF
  __int64 v31; // [rsp+58h] [rbp-58h]
  __int16 v32; // [rsp+70h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v26 = 0;
  v27 = 0;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 40LL))(v2);
  v4 = sub_ECD690(v3);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, const char **))(**(_QWORD **)(a1 + 8) + 192LL))(
         *(_QWORD *)(a1 + 8),
         &v26) )
  {
    goto LABEL_11;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
LABEL_8:
    HIBYTE(v32) = 1;
    v17 = "expected a comma";
LABEL_9:
    v18 = *(_QWORD *)(a1 + 8);
    v30 = v17;
    LOBYTE(v32) = 3;
    return (unsigned int)sub_ECE0E0(v18, (__int64)&v30, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v5 = *(_QWORD *)(a1 + 8);
  v28 = 0;
  v29 = 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5);
  v7 = sub_ECD690(v6);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, const char **))(**(_QWORD **)(a1 + 8) + 192LL))(
         *(_QWORD *)(a1 + 8),
         &v28) )
  {
LABEL_11:
    HIBYTE(v32) = 1;
    v17 = "expected identifier in directive";
    goto LABEL_9;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
    goto LABEL_8;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v8 = *(_QWORD *)(a1 + 8);
  v32 = 259;
  v30 = "expected integer";
  v9 = sub_ECE130(v8, &v25, (__int64)&v30);
  if ( !(_BYTE)v9 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v32 = 261;
      v30 = v26;
      v31 = v27;
      v21 = sub_E6C460(v10, &v30);
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v32 = 261;
      v30 = v28;
      v31 = v29;
      v22 = sub_E6C460(v11, &v30);
      v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v24 = v25;
      v23 = *(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64, __int64))(*(_QWORD *)v12 + 1184LL);
      v13 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v14 = sub_E808D0(v22, 0, v13, v7);
      v15 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v16 = sub_E808D0(v21, 0, v15, v4);
      v23(v12, v16, v14, v24);
    }
    else
    {
      v20 = *(_QWORD *)(a1 + 8);
      v30 = "unexpected token in directive";
      v32 = 259;
      return (unsigned int)sub_ECE0E0(v20, (__int64)&v30, 0, 0);
    }
  }
  return v9;
}
