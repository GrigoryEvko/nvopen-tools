// Function: sub_1B47380
// Address: 0x1b47380
//
__int64 __fastcall sub_1B47380(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r10
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rdi
  unsigned __int8 v21; // al
  __int64 v22; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 v23; // [rsp-C0h] [rbp-C0h]
  int v24; // [rsp-B0h] [rbp-B0h] BYREF
  int v25; // [rsp-ACh] [rbp-ACh] BYREF
  __int64 v26; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v27; // [rsp-A0h] [rbp-A0h] BYREF
  __int64 *v28; // [rsp-98h] [rbp-98h] BYREF
  __int64 *v29; // [rsp-90h] [rbp-90h] BYREF
  __int64 v30; // [rsp-88h] [rbp-88h] BYREF
  _BYTE *v31; // [rsp-80h] [rbp-80h]
  _BYTE *v32; // [rsp-78h] [rbp-78h]
  __int64 v33; // [rsp-70h] [rbp-70h]
  int v34; // [rsp-68h] [rbp-68h]
  _BYTE v35[96]; // [rsp-60h] [rbp-60h] BYREF

  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = sub_1AA6DC0(v3, &v26, &v27);
  if ( !v4 || *(_BYTE *)(v4 + 16) == 13 )
    return 0;
  v5 = *(_QWORD *)(v3 + 48);
  v6 = 0;
  v7 = v5;
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    if ( *(_BYTE *)(v7 - 8) != 77 )
      break;
    v7 = *(_QWORD *)(v7 + 8);
    ++v6;
  }
  if ( v6 > 2 )
    return 0;
  v30 = 0;
  v8 = v5;
  v31 = v35;
  v32 = v35;
  v33 = 4;
  v34 = 0;
  v24 = dword_4FB7680;
  v25 = dword_4FB7680;
  while ( 1 )
  {
    if ( !v8 )
      BUG();
    if ( *(_BYTE *)(v8 - 8) != 77 )
      break;
    v9 = v8 - 24;
    v22 = *(_QWORD *)(v8 + 8);
    v10 = sub_15F2050(a1);
    v11 = sub_1632FA0(v10);
    sub_14A2630(&v28, v11);
    if ( (*(_BYTE *)(v8 - 1) & 0x40) != 0 )
      v12 = *(_QWORD *)(v8 - 32);
    else
      v12 = v9 - 24LL * (*(_DWORD *)(v8 - 4) & 0xFFFFFFF);
    if ( !dword_4FB7060 )
    {
LABEL_18:
      sub_14A3B20(&v28);
      result = 0;
      goto LABEL_19;
    }
    v13 = *(_QWORD *)v12;
    v14 = *(_BYTE *)(*(_QWORD *)v12 + 16LL);
    if ( v14 <= 0x17u )
    {
      if ( v14 == 5 && (unsigned __int8)sub_1593DF0(v13, v11, v12, (_QWORD *)(unsigned int)dword_4FB7060) )
        goto LABEL_18;
    }
    else if ( !(unsigned __int8)sub_1B47110(v13, v3, (__int64)&v30, (unsigned int *)&v24, &v28, 0) )
    {
      goto LABEL_18;
    }
    v15 = sub_15F2050(a1);
    v16 = sub_1632FA0(v15);
    sub_14A2630(&v29, v16);
    if ( (*(_BYTE *)(v8 - 1) & 0x40) != 0 )
      v19 = *(_QWORD *)(v8 - 32);
    else
      v19 = v9 - 24LL * (*(_DWORD *)(v8 - 4) & 0xFFFFFFF);
    if ( !dword_4FB7060 )
    {
LABEL_34:
      sub_14A3B20(&v29);
      sub_14A3B20(&v28);
      result = 0;
      goto LABEL_19;
    }
    v20 = *(_QWORD *)(v19 + 24);
    v21 = *(_BYTE *)(v20 + 16);
    if ( v21 <= 0x17u )
    {
      if ( v21 == 5 && (unsigned __int8)sub_1593DF0(v20, v16, v17, v18) )
        goto LABEL_34;
    }
    else if ( !(unsigned __int8)sub_1B47110(v20, v3, (__int64)&v30, (unsigned int *)&v25, &v29, 0) )
    {
      goto LABEL_34;
    }
    sub_14A3B20(&v29);
    sub_14A3B20(&v28);
    v8 = v22;
  }
  result = 1;
LABEL_19:
  if ( v32 != v31 )
  {
    v23 = result;
    _libc_free((unsigned __int64)v32);
    return v23;
  }
  return result;
}
