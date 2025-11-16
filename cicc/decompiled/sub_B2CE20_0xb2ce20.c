// Function: sub_B2CE20
// Address: 0xb2ce20
//
__int64 __fastcall sub_B2CE20(__int64 a1, char a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r13
  int v9; // eax
  int v10; // eax
  __int64 v11; // rax
  const void *v12; // rax
  size_t v13; // rdx
  __int64 v14; // rax
  const void *v15; // rax
  size_t v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // r15d
  bool v20; // al
  __int64 v21; // rax
  __int64 v22; // rdi
  size_t v23; // r15
  char *v24; // rcx
  int v25; // eax
  bool v26; // al
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned int v29; // r15d
  bool v30; // al
  const char *v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // r15d
  __int64 v37; // rax
  __int64 v38; // rdi
  unsigned int v39; // r15d
  __int64 v41; // rax
  __int64 v42; // rdi
  unsigned int v43; // r12d
  char *v46; // rcx
  char *v47; // [rsp+0h] [rbp-A0h]
  int v49; // [rsp+8h] [rbp-98h]
  __int64 *v50; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v51; // [rsp+18h] [rbp-88h]
  __int64 v52; // [rsp+20h] [rbp-80h]
  _BYTE v53[120]; // [rsp+28h] [rbp-78h] BYREF

  v7 = sub_BD2DA0(136);
  v8 = v7;
  if ( v7 )
    sub_B2C3B0(v7, a1, a2, a3, a4, a5);
  v51 = v53;
  v50 = (__int64 *)sub_B2BE50(v8);
  v52 = 0x800000000LL;
  v9 = sub_BAA810(a5);
  if ( v9 )
  {
    sub_A77CB0(&v50, v9);
    v10 = sub_BAA840(a5);
    if ( v10 != 2 )
      goto LABEL_5;
LABEL_62:
    sub_A78980(&v50, "frame-pointer", 0xDu, "all", 3u);
    goto LABEL_7;
  }
  v10 = sub_BAA840(a5);
  if ( v10 == 2 )
    goto LABEL_62;
LABEL_5:
  if ( v10 == 3 )
  {
    v46 = "reserved";
  }
  else
  {
    if ( v10 != 1 )
      goto LABEL_7;
    v46 = "non-leaf";
  }
  sub_A78980(&v50, "frame-pointer", 0xDu, v46, 8u);
LABEL_7:
  if ( sub_BA91D0(a5, "function_return_thunk_extern", 28) )
    sub_A77B20(&v50, 11);
  v11 = sub_B2BE50(v8);
  v12 = (const void *)sub_B6F980(v11);
  if ( v13 )
    sub_A78980(&v50, "target-cpu", 0xAu, v12, v13);
  v14 = sub_B2BE50(v8);
  v15 = (const void *)sub_B6F9A0(v14);
  if ( v16 )
    sub_A78980(&v50, "target-features", 0xFu, v15, v16);
  v17 = sub_BA91D0(a5, "sign-return-address", 19);
  if ( !v17
    || (v18 = *(_QWORD *)(v17 + 136)) == 0
    || ((v19 = *(_DWORD *)(v18 + 32), v19 <= 0x40)
      ? (v20 = *(_QWORD *)(v18 + 24) == 0)
      : (v20 = v19 == (unsigned int)sub_C444A0(v18 + 24)),
        v20) )
  {
    v32 = sub_BA91D0(a5, "sign-return-address-all", 23);
    if ( !v32 )
      goto LABEL_33;
    v22 = *(_QWORD *)(v32 + 136);
    v23 = 4;
    v24 = "none";
    if ( !v22 )
    {
LABEL_32:
      if ( *(_DWORD *)v24 == 1701736302 )
        goto LABEL_33;
      v23 = 4;
      goto LABEL_24;
    }
LABEL_20:
    if ( *(_DWORD *)(v22 + 32) <= 0x40u )
    {
      v26 = *(_QWORD *)(v22 + 24) == 0;
    }
    else
    {
      v47 = v24;
      v49 = *(_DWORD *)(v22 + 32);
      v25 = sub_C444A0(v22 + 24);
      v24 = v47;
      v26 = v49 == v25;
    }
    if ( !v26 )
    {
      v24 = "all";
      v23 = 3;
      goto LABEL_24;
    }
    if ( v23 != 4 )
      goto LABEL_24;
    goto LABEL_32;
  }
  v21 = sub_BA91D0(a5, "sign-return-address-all", 23);
  if ( !v21 )
  {
    v24 = "non-leaf";
    v23 = 8;
    goto LABEL_24;
  }
  v22 = *(_QWORD *)(v21 + 136);
  v23 = 8;
  v24 = "non-leaf";
  if ( v22 )
    goto LABEL_20;
LABEL_24:
  sub_A78980(&v50, "sign-return-address", 0x13u, v24, v23);
  v27 = sub_BA91D0(a5, "sign-return-address-with-bkey", 29);
  if ( v27
    && (v28 = *(_QWORD *)(v27 + 136)) != 0
    && ((v29 = *(_DWORD *)(v28 + 32), v29 <= 0x40)
      ? (v30 = *(_QWORD *)(v28 + 24) == 0)
      : (v30 = v29 == (unsigned int)sub_C444A0(v28 + 24)),
        !v30) )
  {
    v31 = "b_key";
  }
  else
  {
    v31 = "a_key";
  }
  sub_A78980(&v50, "sign-return-address-key", 0x17u, v31, 5u);
LABEL_33:
  v33 = sub_BA91D0(a5, "branch-target-enforcement", 25);
  if ( v33 )
  {
    v34 = *(_QWORD *)(v33 + 136);
    if ( v34 )
    {
      v35 = *(_DWORD *)(v34 + 32);
      if ( !(v35 <= 0x40 ? *(_QWORD *)(v34 + 24) == 0 : v35 == (unsigned int)sub_C444A0(v34 + 24)) )
        sub_A78980(&v50, "branch-target-enforcement", 0x19u, 0, 0);
    }
  }
  v37 = sub_BA91D0(a5, "branch-protection-pauth-lr", 26);
  if ( v37 )
  {
    v38 = *(_QWORD *)(v37 + 136);
    if ( v38 )
    {
      v39 = *(_DWORD *)(v38 + 32);
      if ( !(v39 <= 0x40 ? *(_QWORD *)(v38 + 24) == 0 : v39 == (unsigned int)sub_C444A0(v38 + 24)) )
        sub_A78980(&v50, "branch-protection-pauth-lr", 0x1Au, 0, 0);
    }
  }
  v41 = sub_BA91D0(a5, "guarded-control-stack", 21);
  if ( v41 )
  {
    v42 = *(_QWORD *)(v41 + 136);
    if ( v42 )
    {
      v43 = *(_DWORD *)(v42 + 32);
      if ( !(v43 <= 0x40 ? *(_QWORD *)(v42 + 24) == 0 : v43 == (unsigned int)sub_C444A0(v42 + 24)) )
        sub_A78980(&v50, "guarded-control-stack", 0x15u, 0, 0);
    }
  }
  sub_B2CDF0(v8, (__int64)&v50);
  if ( v51 != v53 )
    _libc_free(v51, &v50);
  return v8;
}
