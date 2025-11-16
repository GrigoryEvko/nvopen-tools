// Function: sub_3903470
// Address: 0x3903470
//
__int64 __fastcall sub_3903470(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // r15d
  __int64 v14; // rdi
  const char *v15; // rax
  __int64 v16; // rdi
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r12
  void (__fastcall *v21)(__int64, __int64, __int64, __int64, _QWORD); // r14
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // [rsp+0h] [rbp-80h]
  __int64 v30; // [rsp+0h] [rbp-80h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+10h] [rbp-70h] BYREF
  __int64 v33; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v34[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v35[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v36; // [rsp+40h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v3 = sub_3909290(v2);
  v4 = *(_QWORD *)(a1 + 8);
  v34[0] = 0;
  v34[1] = 0;
  v5 = v3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v4 + 144LL))(v4, v34) )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v35[0] = "expected identifier in directive";
    v36 = 259;
    return (unsigned int)sub_3909CF0(v18, v35, 0, 0, v6, v7);
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v35[0] = v34;
  v36 = 261;
  v31 = sub_38BF510(v8, (__int64)v35);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    HIBYTE(v36) = 1;
    v15 = "unexpected token in directive";
    goto LABEL_8;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v12 = sub_3909290(v11);
  v13 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(*(_QWORD *)(a1 + 8), &v32);
  if ( (_BYTE)v13 )
    return 1;
  v14 = *(_QWORD *)(a1 + 8);
  v33 = 0;
  v29 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 40LL))(v14) + 8) == 25 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v29 = sub_3909290(v19);
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(
           *(_QWORD *)(a1 + 8),
           &v33) )
    {
      return 1;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v36) = 1;
    v15 = "unexpected token in '.tbss' directive";
LABEL_8:
    v16 = *(_QWORD *)(a1 + 8);
    v35[0] = v15;
    LOBYTE(v36) = 3;
    return (unsigned int)sub_3909CF0(v16, v35, 0, 0, v9, v10);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( v32 < 0 )
  {
    v26 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid '.tbss' directive size, can't be less thanzero";
    v36 = 259;
    return (unsigned int)sub_3909790(v26, v12, v35, 0, 0);
  }
  else if ( v33 < 0 )
  {
    v25 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid '.tbss' alignment, can't be lessthan zero";
    v36 = 259;
    return (unsigned int)sub_3909790(v25, v29, v35, 0, 0);
  }
  else if ( (*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL) != 0
         || (*(_BYTE *)(v31 + 9) & 0xC) == 8
         && (*(_BYTE *)(v31 + 8) |= 4u,
             v27 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v31 + 24)),
             *(_QWORD *)v31 = v27 | *(_QWORD *)v31 & 7LL,
             v27) )
  {
    v28 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid symbol redefinition";
    v36 = 259;
    return (unsigned int)sub_3909790(v28, v5, v35, 0, 0);
  }
  else
  {
    v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v30 = v32;
    v21 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v20 + 392LL);
    v22 = 1 << v33;
    v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v24 = sub_38BFA90(v23, "__DATA", 6u, "__thread_bss", 0xCu, 18, 0, 11, 0);
    v21(v20, v24, v31, v30, v22);
  }
  return v13;
}
