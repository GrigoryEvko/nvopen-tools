// Function: sub_39B2D50
// Address: 0x39b2d50
//
__int64 __fastcall sub_39B2D50(__int64 a1)
{
  _QWORD *v1; // rcx
  __int64 v2; // rdx
  __int64 (__fastcall *v3)(unsigned __int64 *); // r12
  __int64 v4; // r12
  __int64 (*v5)(void); // rax
  __int64 v6; // rax
  __int64 (__fastcall *v7)(unsigned __int64 *, __int64, __int64, __int64, __int64); // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 (__fastcall *v13)(__int64, unsigned __int64 *); // r12
  __int64 v14; // r14
  __int64 v15; // r12
  __int64 v16; // rax
  void (__fastcall *v17)(__int64, char); // rdx
  void (__fastcall *v18)(__int64, char); // rax
  bool v19; // si
  __int64 result; // rax
  __int64 v21; // [rsp+0h] [rbp-B0h]
  __int64 v22; // [rsp+8h] [rbp-A8h]
  __int64 v23; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v24; // [rsp+18h] [rbp-98h]
  _QWORD v25[2]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v26; // [rsp+30h] [rbp-80h]
  unsigned __int64 v27[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v28[12]; // [rsp+50h] [rbp-60h] BYREF

  v1 = *(_QWORD **)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 472);
  v3 = (__int64 (__fastcall *)(unsigned __int64 *))v1[9];
  v24 = *(_QWORD *)(a1 + 480);
  v23 = v2;
  if ( v3 )
  {
    v26 = 261;
    v25[0] = &v23;
    sub_16E1010((__int64)v27, (__int64)v25);
    v4 = v3(v27);
    if ( (_QWORD *)v27[0] != v28 )
      j_j___libc_free_0(v27[0]);
    v1 = *(_QWORD **)(a1 + 8);
  }
  else
  {
    v4 = 0;
  }
  *(_QWORD *)(a1 + 616) = v4;
  v5 = (__int64 (*)(void))v1[7];
  if ( v5 )
  {
    v6 = v5();
    v1 = *(_QWORD **)(a1 + 8);
  }
  else
  {
    v6 = 0;
  }
  *(_QWORD *)(a1 + 624) = v6;
  v7 = (__int64 (__fastcall *)(unsigned __int64 *, __int64, __int64, __int64, __int64))v1[10];
  v8 = *(_QWORD *)(a1 + 472);
  v9 = *(_QWORD *)(a1 + 480);
  v21 = *(_QWORD *)(a1 + 568);
  v22 = *(_QWORD *)(a1 + 528);
  v10 = *(_QWORD *)(a1 + 560);
  v23 = v8;
  v11 = *(_QWORD *)(a1 + 536);
  v24 = v9;
  if ( v7 )
  {
    v26 = 261;
    v25[0] = &v23;
    sub_16E1010((__int64)v27, (__int64)v25);
    v12 = v7(v27, v22, v11, v10, v21);
    if ( (_QWORD *)v27[0] != v28 )
      j_j___libc_free_0(v27[0]);
    v1 = *(_QWORD **)(a1 + 8);
    v9 = *(_QWORD *)(a1 + 480);
    v8 = *(_QWORD *)(a1 + 472);
  }
  else
  {
    v12 = 0;
  }
  *(_QWORD *)(a1 + 632) = v12;
  v13 = (__int64 (__fastcall *)(__int64, unsigned __int64 *))v1[6];
  v14 = *(_QWORD *)(a1 + 616);
  v23 = v8;
  v24 = v9;
  if ( v13 )
  {
    v26 = 261;
    v25[0] = &v23;
    sub_16E1010((__int64)v27, (__int64)v25);
    v15 = v13(v14, v27);
    if ( (_QWORD *)v27[0] != v28 )
      j_j___libc_free_0(v27[0]);
  }
  else
  {
    v15 = 0;
  }
  v16 = *(_QWORD *)v15;
  if ( (*(_BYTE *)(a1 + 800) & 0x10) != 0 )
  {
    v17 = *(void (__fastcall **)(__int64, char))(v16 + 64);
    if ( v17 == sub_21BC3C0 )
    {
      *(_BYTE *)(v15 + 392) = 0;
    }
    else
    {
      v17(v15, 0);
      v16 = *(_QWORD *)v15;
    }
  }
  v18 = *(void (__fastcall **)(__int64, char))(v16 + 72);
  v19 = (*(_BYTE *)(a1 + 841) & 0x20) != 0;
  if ( v18 == sub_21BC3D0 )
    *(_BYTE *)(v15 + 393) = v19;
  else
    v18(v15, v19);
  *(_DWORD *)(v15 + 396) = *(_DWORD *)(a1 + 804);
  *(_BYTE *)(v15 + 401) = *(_BYTE *)(a1 + 808) & 1;
  result = *(unsigned int *)(a1 + 836);
  if ( (_DWORD)result )
    *(_DWORD *)(v15 + 348) = result;
  if ( (*(_BYTE *)(a1 + 809) & 0x20) != 0 )
    *(_DWORD *)(v15 + 348) = 0;
  *(_QWORD *)(a1 + 608) = v15;
  return result;
}
