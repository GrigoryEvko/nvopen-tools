// Function: sub_17296C0
// Address: 0x17296c0
//
__int64 __fastcall sub_17296C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v8; // rax
  unsigned int v12; // eax
  __int64 v13; // r8
  bool v14; // al
  __int64 v15; // r13
  __int64 v16; // r13
  bool v17; // al
  __int64 v18; // rdi
  __int64 *v19; // r13
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 v27; // [rsp-A0h] [rbp-A0h]
  bool v28; // [rsp-98h] [rbp-98h]
  bool v29; // [rsp-98h] [rbp-98h]
  unsigned int v30; // [rsp-90h] [rbp-90h]
  __int64 *v31; // [rsp-90h] [rbp-90h]
  __int64 v32; // [rsp-88h] [rbp-88h]
  const void **v33; // [rsp-80h] [rbp-80h]
  bool v34; // [rsp-80h] [rbp-80h]
  __int64 v35; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v36; // [rsp-70h] [rbp-70h]
  __int64 v37; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v38; // [rsp-60h] [rbp-60h]
  __int64 v39; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v40; // [rsp-50h] [rbp-50h]
  __int16 v41; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)(a2 + 16) != 35 )
    return 0;
  v8 = *(_QWORD *)(a2 + 8);
  if ( !v8 || *(_QWORD *)(v8 + 8) )
    return 0;
  v32 = *(_QWORD *)(a2 - 48);
  v33 = (const void **)(a4 + 24);
  if ( *(_DWORD *)(a4 + 32) > 0x40u )
  {
    v30 = *(_DWORD *)(a4 + 32);
    if ( (unsigned int)sub_16A5940((__int64)v33) == 1 )
    {
      v36 = v30;
      v31 = (__int64 *)(a3 + 24);
      sub_16A4FD0((__int64)&v35, v33);
      goto LABEL_7;
    }
    return 0;
  }
  v21 = *(_QWORD *)(a4 + 24);
  if ( !v21 || (v21 & (v21 - 1)) != 0 )
    return 0;
  v36 = *(_DWORD *)(a4 + 32);
  v31 = (__int64 *)(a3 + 24);
  v35 = v21;
LABEL_7:
  sub_16A7800((__int64)&v35, 1u);
  v12 = v36;
  v36 = 0;
  v38 = v12;
  v37 = v35;
  if ( v12 <= 0x40 )
  {
    v13 = *(_QWORD *)(a3 + 24) & v35;
LABEL_9:
    v14 = v13 == 0;
    goto LABEL_10;
  }
  sub_16A8890(&v37, v31);
  v22 = v38;
  v13 = v37;
  v38 = 0;
  v40 = v22;
  v39 = v37;
  if ( v22 <= 0x40 )
    goto LABEL_9;
  v27 = v37;
  v14 = v22 == (unsigned int)sub_16A57B0((__int64)&v39);
  if ( v27 )
  {
    v29 = v14;
    j_j___libc_free_0_0(v27);
    v14 = v29;
    if ( v38 > 0x40 )
    {
      if ( v37 )
      {
        j_j___libc_free_0_0(v37);
        v14 = v29;
      }
    }
  }
LABEL_10:
  if ( v36 > 0x40 && v35 )
  {
    v28 = v14;
    j_j___libc_free_0_0(v35);
    v14 = v28;
  }
  if ( !v14 )
    return 0;
  v38 = *(_DWORD *)(a3 + 32);
  if ( v38 <= 0x40 )
  {
    v15 = *(_QWORD *)(a3 + 24);
LABEL_16:
    v16 = *(_QWORD *)(a4 + 24) & v15;
    goto LABEL_17;
  }
  sub_16A4FD0((__int64)&v37, (const void **)v31);
  if ( v38 <= 0x40 )
  {
    v15 = v37;
    goto LABEL_16;
  }
  sub_16A8890(&v37, (__int64 *)v33);
  v26 = v38;
  v16 = v37;
  v38 = 0;
  v40 = v26;
  v39 = v37;
  if ( v26 > 0x40 )
  {
    v17 = v26 == (unsigned int)sub_16A57B0((__int64)&v39);
    if ( v16 )
    {
      v34 = v17;
      j_j___libc_free_0_0(v16);
      v17 = v34;
      if ( v38 > 0x40 )
      {
        if ( v37 )
        {
          j_j___libc_free_0_0(v37);
          v17 = v34;
        }
      }
    }
    goto LABEL_18;
  }
LABEL_17:
  v17 = v16 == 0;
LABEL_18:
  if ( v17 )
  {
    if ( *(_QWORD *)(a5 - 48) )
    {
      v23 = *(_QWORD *)(a5 - 40);
      v24 = *(_QWORD *)(a5 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v24 = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
    }
    *(_QWORD *)(a5 - 48) = v32;
    if ( v32 )
    {
      v25 = *(_QWORD *)(v32 + 8);
      *(_QWORD *)(a5 - 40) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = (a5 - 40) | *(_QWORD *)(v25 + 16) & 3LL;
      *(_QWORD *)(a5 - 32) = (v32 + 8) | *(_QWORD *)(a5 - 32) & 3LL;
      *(_QWORD *)(v32 + 8) = a5 - 48;
    }
    return a5;
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 8);
    v41 = 257;
    v19 = (__int64 *)sub_1729500(v18, (unsigned __int8 *)v32, a4, &v39, a6, a7, a8);
    sub_164B7C0((__int64)v19, a2);
    v41 = 257;
    return sub_15FB440(28, v19, a4, (__int64)&v39, 0);
  }
}
