// Function: sub_258A7E0
// Address: 0x258a7e0
//
__int64 __fastcall sub_258A7E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // r12
  __int64 v9; // rsi
  _DWORD *v10; // rdi
  __int64 (*v11)(void); // rax
  int v13; // eax
  __int64 v14; // r13
  bool v15; // cc
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // [rsp-10h] [rbp-B0h]
  const void *v23; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v24; // [rsp+8h] [rbp-98h]
  const void *v25; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-88h]
  void *v27; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-78h]
  const void *v29; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-68h]
  const void *v31; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-58h]
  __int64 v33; // [rsp+50h] [rbp-50h] BYREF
  int v34; // [rsp+58h] [rbp-48h]
  __int64 v35; // [rsp+60h] [rbp-40h]
  int v36; // [rsp+68h] [rbp-38h]

  v3 = sub_250D2C0(a2, **(_QWORD **)a1);
  v5 = sub_2589400(*(_QWORD *)(a1 + 8), v3, v4, *(_QWORD *)(a1 + 16), 0, 0, 1);
  if ( !v5 )
    return 0;
  v6 = v5;
  v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
  if ( v7 == sub_2534AC0 )
    v8 = v6 + 88;
  else
    v8 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v7)(v6, v3, v22);
  v9 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v9 + 80) )
  {
    sub_AADB10((__int64)&v23, *(_DWORD *)(v8 + 8), 0);
    v14 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)(v14 + 80) )
    {
      v27 = &unk_4A16D38;
      v28 = v24;
      v30 = v24;
      if ( v24 > 0x40 )
        sub_C43780((__int64)&v29, &v23);
      else
        v29 = v23;
      v32 = v26;
      if ( v26 > 0x40 )
        sub_C43780((__int64)&v31, &v25);
      else
        v31 = v25;
      sub_AADB10((__int64)&v33, v24, 1);
      v15 = *(_DWORD *)(v14 + 24) <= 0x40u;
      *(_DWORD *)(v14 + 8) = v28;
      if ( !v15 )
      {
        v16 = *(_QWORD *)(v14 + 16);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      *(_QWORD *)(v14 + 16) = v29;
      *(_DWORD *)(v14 + 24) = v30;
      v30 = 0;
      if ( *(_DWORD *)(v14 + 40) > 0x40u )
      {
        v17 = *(_QWORD *)(v14 + 32);
        if ( v17 )
          j_j___libc_free_0_0(v17);
      }
      *(_QWORD *)(v14 + 32) = v31;
      *(_DWORD *)(v14 + 40) = v32;
      v32 = 0;
      if ( *(_DWORD *)(v14 + 56) > 0x40u )
      {
        v18 = *(_QWORD *)(v14 + 48);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      *(_QWORD *)(v14 + 48) = v33;
      *(_DWORD *)(v14 + 56) = v34;
      v34 = 0;
      if ( *(_DWORD *)(v14 + 72) > 0x40u )
      {
        v19 = *(_QWORD *)(v14 + 64);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
      *(_QWORD *)(v14 + 64) = v35;
      *(_DWORD *)(v14 + 72) = v36;
      v36 = 0;
      sub_253FFA0((__int64)&v27);
    }
    else
    {
      *(_QWORD *)v14 = &unk_4A16D38;
      v20 = v24;
      *(_DWORD *)(v14 + 8) = v24;
      *(_DWORD *)(v14 + 24) = v20;
      if ( v20 > 0x40 )
        sub_C43780(v14 + 16, &v23);
      else
        *(_QWORD *)(v14 + 16) = v23;
      v21 = v26;
      *(_DWORD *)(v14 + 40) = v26;
      if ( v21 > 0x40 )
        sub_C43780(v14 + 32, &v25);
      else
        *(_QWORD *)(v14 + 32) = v25;
      sub_AADB10(v14 + 48, v24, 1);
      *(_BYTE *)(v14 + 80) = 1;
    }
    sub_969240((__int64 *)&v25);
    sub_969240((__int64 *)&v23);
    v9 = *(_QWORD *)(a1 + 24);
  }
  sub_254FA20((__int64)&v27, v9, v8);
  sub_253FFA0((__int64)&v27);
  v10 = *(_DWORD **)(a1 + 24);
  v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 16LL);
  if ( (char *)v11 != (char *)sub_2535A50 )
    return v11();
  if ( !v10[2] )
    return 0;
  LOBYTE(v13) = sub_AAF760((__int64)(v10 + 4));
  return v13 ^ 1u;
}
