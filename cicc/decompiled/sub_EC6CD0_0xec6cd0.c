// Function: sub_EC6CD0
// Address: 0xec6cd0
//
__int64 __fastcall sub_EC6CD0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // r15d
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  const char *v15; // rax
  __int64 v16; // rdi
  const char *v17; // rax
  __int64 v18; // rdi
  __int64 v20; // r12
  void (__fastcall *v21)(__int64, unsigned __int64, _QWORD, _QWORD, _QWORD, __int64); // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdi
  void *v28; // rax
  __int64 v29; // r12
  int v30; // ecx
  unsigned __int64 v31; // rax
  unsigned int v32; // r14d
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-B8h]
  void (__fastcall *v36)(__int64, unsigned __int64, __int64, __int64, _QWORD, __int64); // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+10h] [rbp-B0h]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+28h] [rbp-98h] BYREF
  __int64 v42; // [rsp+30h] [rbp-90h] BYREF
  __int64 v43; // [rsp+38h] [rbp-88h]
  void *v44; // [rsp+40h] [rbp-80h] BYREF
  void *v45; // [rsp+48h] [rbp-78h]
  const char *v46; // [rsp+50h] [rbp-70h] BYREF
  const char *v47; // [rsp+58h] [rbp-68h]
  const char *v48[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v49; // [rsp+80h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v42 = 0;
  v43 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v2 + 192LL))(v2, &v42) )
  {
    HIBYTE(v49) = 1;
    v17 = "expected segment name after '.zerofill' directive";
    goto LABEL_14;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
    goto LABEL_13;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v3 = *(_QWORD *)(a1 + 8);
  v44 = 0;
  v45 = 0;
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 40LL))(v3);
  v5 = sub_ECD690(v4);
  v6 = (*(__int64 (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &v44);
  if ( (_BYTE)v6 )
  {
    HIBYTE(v49) = 1;
    v17 = "expected section name after comma in '.zerofill' directive";
    goto LABEL_14;
  }
  v7 = **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9;
  v8 = **(_QWORD **)(a1 + 8);
  if ( v7 )
  {
    v20 = (*(__int64 (**)(void))(v8 + 56))();
    v21 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD, _QWORD, _QWORD, __int64))(*(_QWORD *)v20 + 496LL);
    v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v23 = sub_E6D970(v22, v42, v43, v44, v45, 1, 0, 15, 0);
    v21(v20, v23, 0, 0, 0, v5);
    return v6;
  }
  if ( **(_DWORD **)((*(__int64 (**)(void))(v8 + 40))() + 8) != 26 )
  {
LABEL_13:
    HIBYTE(v49) = 1;
    v17 = "unexpected token in directive";
LABEL_14:
    v18 = *(_QWORD *)(a1 + 8);
    v48[0] = v17;
    LOBYTE(v49) = 3;
    return (unsigned int)sub_ECE0E0(v18, v48, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v10 = sub_ECD690(v9);
  v11 = *(_QWORD *)(a1 + 8);
  v46 = 0;
  v47 = 0;
  v12 = v10;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v11 + 192LL))(v11, &v46) )
  {
    HIBYTE(v49) = 1;
    v17 = "expected identifier in directive";
    goto LABEL_14;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v49 = 261;
  v48[0] = v46;
  v48[1] = v47;
  v39 = sub_E6C460(v13, v48);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
    HIBYTE(v49) = 1;
    v15 = "unexpected token in directive";
    goto LABEL_12;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v37 = sub_ECD690(v14);
  v6 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(*(_QWORD *)(a1 + 8), &v40);
  if ( (_BYTE)v6 )
    return 1;
  v41 = 0;
  v35 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v24 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v35 = sub_ECD690(v24);
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(
           *(_QWORD *)(a1 + 8),
           &v41) )
    {
      return 1;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v49) = 1;
    v15 = "unexpected token in '.zerofill' directive";
LABEL_12:
    v16 = *(_QWORD *)(a1 + 8);
    v48[0] = v15;
    LOBYTE(v49) = 3;
    return (unsigned int)sub_ECE0E0(v16, v48, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( v40 < 0 )
  {
    v27 = *(_QWORD *)(a1 + 8);
    v48[0] = "invalid '.zerofill' directive size, can't be less than zero";
    v49 = 259;
    return (unsigned int)sub_ECDA70(v27, v37, v48, 0, 0);
  }
  else if ( v41 < 0 )
  {
    v26 = *(_QWORD *)(a1 + 8);
    v48[0] = "invalid '.zerofill' directive alignment, can't be less than zero";
    v49 = 259;
    return (unsigned int)sub_ECDA70(v26, v35, v48, 0, 0);
  }
  else if ( *(_QWORD *)v39
         || (*(_BYTE *)(v39 + 9) & 0x70) == 0x20
         && *(char *)(v39 + 8) >= 0
         && (*(_BYTE *)(v39 + 8) |= 8u, v28 = sub_E807D0(*(_QWORD *)(v39 + 24)), (*(_QWORD *)v39 = v28) != 0) )
  {
    v25 = *(_QWORD *)(a1 + 8);
    v48[0] = "invalid symbol redefinition";
    v49 = 259;
    return (unsigned int)sub_ECDA70(v25, v12, v48, 0, 0);
  }
  else
  {
    v29 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v30 = 64;
    if ( 1LL << v41 )
    {
      _BitScanReverse64(&v31, 1LL << v41);
      v30 = v31 ^ 0x3F;
    }
    v36 = *(void (__fastcall **)(__int64, unsigned __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v29 + 496LL);
    v32 = 63 - v30;
    v38 = v40;
    v33 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v34 = sub_E6D970(v33, v42, v43, v44, v45, 1, 0, 15, 0);
    v36(v29, v34, v39, v38, v32, v5);
  }
  return v6;
}
