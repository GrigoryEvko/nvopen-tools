// Function: sub_34CE050
// Address: 0x34ce050
//
__int64 __fastcall sub_34CE050(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, char a5)
{
  _QWORD *v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 (__fastcall *v15)(_QWORD, __int64); // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 (__fastcall *v21)(__int64, __int64, __int64, __int64); // rax
  _DWORD *v22; // rax
  __int64 v23; // r10
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 (__fastcall *v28)(__int64, __int64 *, __int64); // rax
  __int64 v29; // rsi
  unsigned int v30; // r12d
  void (__fastcall *v31)(__int64, __int64, _QWORD); // rbx
  __int64 v32; // rax
  __int64 v34; // [rsp-8h] [rbp-78h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  _DWORD *v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v40; // [rsp+20h] [rbp-50h] BYREF
  _DWORD *v41; // [rsp+28h] [rbp-48h] BYREF
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  __int64 v43[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = (_QWORD *)sub_22077B0(0xAB0u);
  v10 = (__int64)v9;
  if ( v9 )
    sub_2EAA600(v9, (__int64 *)a1);
  if ( !sub_34CD400(a1, a2, a5, v10) )
    return 1;
  v11 = *(_QWORD *)(v10 + 2656);
  if ( !v11 )
    v11 = v10 + 184;
  *a3 = v11;
  v12 = *(_QWORD *)(a1 + 8);
  v13 = *(_QWORD *)(a1 + 664);
  v14 = *(_QWORD *)(a1 + 680);
  *(_DWORD *)(a1 + 992) = 0;
  v15 = *(__int64 (__fastcall **)(_QWORD, __int64))(v12 + 144);
  v16 = *a3;
  v39 = v13;
  if ( !v15 )
    return 1;
  v18 = v15(*(_QWORD *)(a1 + 672), v16);
  if ( v18 )
  {
    v19 = *(_QWORD *)(a1 + 8);
    v20 = a1 + 976;
    v21 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(v19 + 104);
    if ( v21 && (v35 = v18, v16 = v14, v22 = (_DWORD *)v21(v19, v14, v39, v20), v18 = v35, v22) )
    {
      v23 = *(_QWORD *)(a1 + 8);
      v24 = a4;
      v43[0] = v35;
      v38 = v22;
      v36 = v23;
      sub_106DB90(&v42, v22, v24);
      v25 = *a3;
      v26 = a1 + 512;
      v41 = v38;
      v40 = sub_C0D2B0(v36, a1 + 512, v25, (__int64)&v41, (__int64)&v42, (__int64)v43, v14);
      v27 = v34;
      if ( v41 )
        (*(void (__fastcall **)(_DWORD *, __int64))(*(_QWORD *)v41 + 8LL))(v41, v26);
      if ( v42 )
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v42 + 8LL))(v42, v26, v27);
      if ( v43[0] )
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v43[0] + 8LL))(v43[0], v26, v27);
      v28 = *(__int64 (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)(a1 + 8) + 120LL);
      if ( v28 && (v29 = v28(a1, &v40, v27)) != 0 )
      {
        v30 = 0;
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL))(a2, v29, 0);
        v31 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
        v32 = sub_2EAA4A0();
        v31(a2, v32, 0);
      }
      else
      {
        v30 = 1;
      }
      if ( v40 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 56LL))(v40);
    }
    else
    {
      v30 = 1;
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v18 + 8LL))(v18, v16, v17, v20);
    }
  }
  else
  {
    return 1;
  }
  return v30;
}
