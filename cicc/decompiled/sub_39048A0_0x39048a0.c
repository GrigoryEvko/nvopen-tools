// Function: sub_39048A0
// Address: 0x39048a0
//
__int64 __fastcall sub_39048A0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned int v8; // r15d
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  const char *v19; // rax
  __int64 v20; // rdi
  const char *v21; // rax
  __int64 v22; // rdi
  __int64 v24; // r12
  void (__fastcall *v25)(__int64, __int64, _QWORD, _QWORD, _QWORD, __int64); // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  void (__fastcall *v30)(__int64, __int64, __int64, __int64, _QWORD, __int64); // r13
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdi
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // [rsp+8h] [rbp-A8h]
  unsigned int v38; // [rsp+8h] [rbp-A8h]
  __int64 v39; // [rsp+10h] [rbp-A0h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  __int64 v42; // [rsp+20h] [rbp-90h] BYREF
  __int64 v43; // [rsp+28h] [rbp-88h] BYREF
  const void *v44; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v45; // [rsp+38h] [rbp-78h]
  const void *v46; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v47; // [rsp+48h] [rbp-68h]
  _QWORD v48[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v49[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v50; // [rsp+70h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v44 = 0;
  v45 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const void **))(*(_QWORD *)v2 + 144LL))(v2, &v44) )
  {
    HIBYTE(v50) = 1;
    v21 = "expected segment name after '.zerofill' directive";
    goto LABEL_14;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    goto LABEL_13;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v5 = *(_QWORD *)(a1 + 8);
  v46 = 0;
  v47 = 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5);
  v7 = sub_3909290(v6);
  v8 = (*(__int64 (__fastcall **)(_QWORD, const void **))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), &v46);
  if ( (_BYTE)v8 )
  {
    HIBYTE(v50) = 1;
    v21 = "expected section name after comma in '.zerofill' directive";
    goto LABEL_14;
  }
  v9 = **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9;
  v10 = **(_QWORD **)(a1 + 8);
  if ( v9 )
  {
    v24 = (*(__int64 (**)(void))(v10 + 56))();
    v25 = *(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD, __int64))(*(_QWORD *)v24 + 384LL);
    v26 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v27 = sub_38BFA90(v26, v44, v45, v46, v47, 1, 0, 13, 0);
    v25(v24, v27, 0, 0, 0, v7);
    return v8;
  }
  if ( **(_DWORD **)((*(__int64 (**)(void))(v10 + 40))() + 8) != 25 )
  {
LABEL_13:
    HIBYTE(v50) = 1;
    v21 = "unexpected token in directive";
LABEL_14:
    v22 = *(_QWORD *)(a1 + 8);
    v49[0] = v21;
    LOBYTE(v50) = 3;
    return (unsigned int)sub_3909CF0(v22, v49, 0, 0, v3, v4);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v12 = sub_3909290(v11);
  v13 = *(_QWORD *)(a1 + 8);
  v48[0] = 0;
  v48[1] = 0;
  v14 = v12;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v13 + 144LL))(v13, v48) )
  {
    HIBYTE(v50) = 1;
    v21 = "expected identifier in directive";
    goto LABEL_14;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v49[0] = v48;
  v50 = 261;
  v41 = sub_38BF510(v15, (__int64)v49);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    HIBYTE(v50) = 1;
    v19 = "unexpected token in directive";
    goto LABEL_12;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v18 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v39 = sub_3909290(v18);
  v8 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(*(_QWORD *)(a1 + 8), &v42);
  if ( (_BYTE)v8 )
    return 1;
  v43 = 0;
  v37 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v28 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v37 = sub_3909290(v28);
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(
           *(_QWORD *)(a1 + 8),
           &v43) )
    {
      return 1;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v50) = 1;
    v19 = "unexpected token in '.zerofill' directive";
LABEL_12:
    v20 = *(_QWORD *)(a1 + 8);
    v49[0] = v19;
    LOBYTE(v50) = 3;
    return (unsigned int)sub_3909CF0(v20, v49, 0, 0, v16, v17);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( v42 < 0 )
  {
    v34 = *(_QWORD *)(a1 + 8);
    v49[0] = "invalid '.zerofill' directive size, can't be less than zero";
    v50 = 259;
    return (unsigned int)sub_3909790(v34, v39, v49, 0, 0);
  }
  else if ( v43 < 0 )
  {
    v33 = *(_QWORD *)(a1 + 8);
    v49[0] = "invalid '.zerofill' directive alignment, can't be less than zero";
    v50 = 259;
    return (unsigned int)sub_3909790(v33, v37, v49, 0, 0);
  }
  else if ( (*(_QWORD *)v41 & 0xFFFFFFFFFFFFFFF8LL) != 0
         || (*(_BYTE *)(v41 + 9) & 0xC) == 8
         && (*(_BYTE *)(v41 + 8) |= 4u,
             v35 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v41 + 24)),
             *(_QWORD *)v41 = v35 | *(_QWORD *)v41 & 7LL,
             v35) )
  {
    v36 = *(_QWORD *)(a1 + 8);
    v49[0] = "invalid symbol redefinition";
    v50 = 259;
    return (unsigned int)sub_3909790(v36, v14, v49, 0, 0);
  }
  else
  {
    v29 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v30 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v29 + 384LL);
    v40 = v42;
    v38 = 1 << v43;
    v31 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v32 = sub_38BFA90(v31, v44, v45, v46, v47, 1, 0, 13, 0);
    v30(v29, v32, v41, v40, v38, v7);
  }
  return v8;
}
