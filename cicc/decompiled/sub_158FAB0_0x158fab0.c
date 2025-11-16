// Function: sub_158FAB0
// Address: 0x158fab0
//
__int64 __fastcall sub_158FAB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r15d
  bool v7; // r15
  unsigned int v8; // ecx
  int v9; // eax
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 v12; // r13
  unsigned int v13; // edx
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdi
  int v17; // eax
  unsigned int v18; // ecx
  unsigned int v19; // [rsp+8h] [rbp-98h]
  unsigned int v20; // [rsp+Ch] [rbp-94h]
  unsigned int v21; // [rsp+Ch] [rbp-94h]
  unsigned int v22; // [rsp+Ch] [rbp-94h]
  unsigned int v23; // [rsp+18h] [rbp-88h]
  unsigned int v24; // [rsp+18h] [rbp-88h]
  unsigned int v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-78h]
  _QWORD *v28; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-68h]
  __int64 v30; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-58h]
  __int64 v32; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-48h]
  __int64 v34; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
    goto LABEL_2;
  sub_158A9F0((__int64)&v34, a3);
  v6 = v35;
  if ( v35 <= 0x40 )
  {
    v7 = v34 == 0;
  }
  else
  {
    v7 = v6 == (unsigned int)sub_16A57B0(&v34);
    if ( v34 )
      j_j___libc_free_0_0(v34);
  }
  if ( v7 )
  {
LABEL_2:
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  if ( sub_158A0B0(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
    return a1;
  }
  sub_158AAD0((__int64)&v32, a2);
  sub_158A9F0((__int64)&v34, a3);
  sub_16A9D70(&v26, &v32, &v34);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  sub_158AAD0((__int64)&v28, a3);
  v8 = v29;
  if ( v29 <= 0x40 )
  {
    if ( v28 )
      goto LABEL_24;
  }
  else
  {
    v20 = v29;
    v9 = sub_16A57B0(&v28);
    v8 = v20;
    if ( v20 != v9 )
      goto LABEL_24;
  }
  v21 = *(_DWORD *)(a3 + 24);
  if ( v21 > 0x40 )
  {
    v19 = v8;
    v17 = sub_16A57B0(a3 + 16);
    v8 = v19;
    if ( v21 - v17 > 0x40 )
    {
LABEL_22:
      if ( v8 > 0x40 )
      {
        *v28 = 1;
        memset(v28 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v29 + 63) >> 6) - 8);
      }
      else
      {
        v28 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & 1);
      }
      goto LABEL_24;
    }
    v10 = **(_QWORD **)(a3 + 16);
  }
  else
  {
    v10 = *(_QWORD *)(a3 + 16);
  }
  if ( v10 != 1 )
    goto LABEL_22;
  if ( v8 <= 0x40 && (v18 = *(_DWORD *)(a3 + 8), v18 <= 0x40) )
  {
    v29 = *(_DWORD *)(a3 + 8);
    v28 = (_QWORD *)(*(_QWORD *)a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18));
  }
  else
  {
    sub_16A51C0(&v28, a3);
  }
LABEL_24:
  sub_158A9F0((__int64)&v32, a2);
  sub_16A9D70(&v34, &v32, &v28);
  sub_16A7490(&v34, 1);
  v11 = v35;
  v12 = v34;
  v31 = v35;
  v30 = v34;
  if ( v33 > 0x40 && v32 )
  {
    v23 = v35;
    j_j___libc_free_0_0(v32);
    v11 = v23;
  }
  v13 = v27;
  if ( v27 <= 0x40 )
  {
    v15 = v26;
    if ( v12 != v26 )
      goto LABEL_30;
  }
  else
  {
    v24 = v11;
    v22 = v27;
    v14 = sub_16A5220(&v26, &v30);
    v11 = v24;
    if ( !v14 )
    {
      v15 = v26;
      v13 = v22;
LABEL_30:
      v33 = v13;
      v35 = v11;
      v34 = v12;
      v32 = v15;
      v27 = 0;
      sub_15898E0(a1, (__int64)&v32, &v34);
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      if ( v35 <= 0x40 )
        goto LABEL_36;
      v16 = v34;
      if ( !v34 )
        goto LABEL_36;
      goto LABEL_35;
    }
  }
  v25 = v11;
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  if ( v25 <= 0x40 || !v12 )
    goto LABEL_36;
  v16 = v12;
LABEL_35:
  j_j___libc_free_0_0(v16);
LABEL_36:
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return a1;
}
