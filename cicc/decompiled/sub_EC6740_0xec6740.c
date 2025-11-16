// Function: sub_EC6740
// Address: 0xec6740
//
__int64 __fastcall sub_EC6740(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // r15d
  __int64 v10; // rdi
  const char *v11; // rax
  __int64 v12; // rdi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  void *v19; // rax
  __int64 v20; // rax
  char v21; // r11
  __int64 v22; // r12
  void (__fastcall *v23)(__int64, unsigned __int64, __int64, __int64, _QWORD); // r14
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-90h]
  unsigned __int8 v28; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h] BYREF
  __int64 v32; // [rsp+18h] [rbp-78h] BYREF
  const char *v33; // [rsp+20h] [rbp-70h] BYREF
  const char *v34; // [rsp+28h] [rbp-68h]
  const char *v35[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v36; // [rsp+50h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v3 = sub_ECD690(v2);
  v4 = *(_QWORD *)(a1 + 8);
  v33 = 0;
  v34 = 0;
  v5 = v3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v4 + 192LL))(v4, &v33) )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v35[0] = "expected identifier in directive";
    v36 = 259;
    return (unsigned int)sub_ECE0E0(v14, v35, 0, 0);
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v36 = 261;
  v35[0] = v33;
  v35[1] = v34;
  v7 = sub_E6C460(v6, v35);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
    HIBYTE(v36) = 1;
    v11 = "unexpected token in directive";
    goto LABEL_8;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v29 = sub_ECD690(v8);
  v9 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(*(_QWORD *)(a1 + 8), &v31);
  if ( (_BYTE)v9 )
    return 1;
  v10 = *(_QWORD *)(a1 + 8);
  v32 = 0;
  v27 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10) + 8) == 26 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v27 = sub_ECD690(v15);
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(
           *(_QWORD *)(a1 + 8),
           &v32) )
    {
      return 1;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v36) = 1;
    v11 = "unexpected token in '.tbss' directive";
LABEL_8:
    v12 = *(_QWORD *)(a1 + 8);
    v35[0] = v11;
    LOBYTE(v36) = 3;
    return (unsigned int)sub_ECE0E0(v12, v35, 0, 0);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( v31 < 0 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid '.tbss' directive size, can't be less thanzero";
    v36 = 259;
    return (unsigned int)sub_ECDA70(v18, v29, v35, 0, 0);
  }
  else if ( v32 < 0 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid '.tbss' alignment, can't be lessthan zero";
    v36 = 259;
    return (unsigned int)sub_ECDA70(v17, v27, v35, 0, 0);
  }
  else if ( *(_QWORD *)v7
         || (*(_BYTE *)(v7 + 9) & 0x70) == 0x20
         && *(char *)(v7 + 8) >= 0
         && (*(_BYTE *)(v7 + 8) |= 8u, v19 = sub_E807D0(*(_QWORD *)(v7 + 24)), (*(_QWORD *)v7 = v19) != 0) )
  {
    v16 = *(_QWORD *)(a1 + 8);
    v35[0] = "invalid symbol redefinition";
    v36 = 259;
    return (unsigned int)sub_ECDA70(v16, v5, v35, 0, 0);
  }
  else
  {
    v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v21 = -1;
    v22 = v20;
    v23 = *(void (__fastcall **)(__int64, unsigned __int64, __int64, __int64, _QWORD))(*(_QWORD *)v20 + 504LL);
    if ( 1LL << v32 )
    {
      _BitScanReverse64(&v24, 1LL << v32);
      v21 = 63 - (v24 ^ 0x3F);
    }
    v28 = v21;
    v30 = v31;
    v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v26 = sub_E6D970(v25, (__int64)"__DATA", 6, "__thread_bss", (void *)0xC, 18, 0, 12, 0);
    v23(v22, v26, v7, v30, v28);
  }
  return v9;
}
