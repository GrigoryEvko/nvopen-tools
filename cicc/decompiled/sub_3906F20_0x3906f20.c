// Function: sub_3906F20
// Address: 0x3906f20
//
__int64 __fastcall sub_3906F20(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned int v14; // r9d
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r14
  void (__fastcall *v20)(__int64, __int64, __int64, __int64); // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  const char *v24; // rax
  __int64 v25; // rdi
  __int64 v27; // rdi
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v33[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v34[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v35[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v33[0] = 0;
  v33[1] = 0;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 40LL))(v2);
  v4 = sub_3909290(v3);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD *))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), v33) )
    goto LABEL_11;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
LABEL_8:
    HIBYTE(v36) = 1;
    v24 = "expected a comma";
LABEL_9:
    v25 = *(_QWORD *)(a1 + 8);
    v35[0] = v24;
    LOBYTE(v36) = 3;
    return (unsigned int)sub_3909CF0(v25, v35, 0, 0, v5, v6);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v7 = *(_QWORD *)(a1 + 8);
  v34[0] = 0;
  v34[1] = 0;
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7);
  v30 = sub_3909290(v8);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD *))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), v34) )
  {
LABEL_11:
    HIBYTE(v36) = 1;
    v24 = "expected identifier in directive";
    goto LABEL_9;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    goto LABEL_8;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v9 = *(_QWORD *)(a1 + 8);
  v35[0] = "expected integer count in '.cg_profile' directive";
  v36 = 259;
  v10 = sub_3909D40(v9, &v32, v35);
  v14 = v10;
  if ( !(_BYTE)v10 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(
                         *(_QWORD *)(a1 + 8),
                         &v32,
                         v11,
                         v12,
                         v13,
                         v10)
                     + 8) == 9 )
    {
      v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v35[0] = v33;
      v36 = 261;
      v17 = sub_38BF510(v16, (__int64)v35);
      v18 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v35[0] = v34;
      v36 = 261;
      v28 = sub_38BF510(v18, (__int64)v35);
      v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v29 = v32;
      v20 = *(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v19 + 976LL);
      v21 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v31 = sub_38CF310(v28, 0, v21, v30);
      v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v23 = sub_38CF310(v17, 0, v22, v4);
      v20(v19, v23, v31, v29);
      return 0;
    }
    else
    {
      v27 = *(_QWORD *)(a1 + 8);
      v35[0] = "unexpected token in directive";
      v36 = 259;
      return (unsigned int)sub_3909CF0(v27, v35, 0, 0, v15, 0);
    }
  }
  return v14;
}
