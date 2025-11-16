// Function: sub_16AECF0
// Address: 0x16aecf0
//
__int64 __fastcall sub_16AECF0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned int v6; // r14d
  unsigned int v7; // ecx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned int v10; // esi
  bool v11; // zf
  __int64 v12; // rdx
  unsigned int v13; // ecx
  const void *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // esi
  bool v17; // zf
  __int64 v18; // rdx
  unsigned int v19; // eax
  const void *v21; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-58h]
  _QWORD *v23; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-48h]
  const void *v25; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-38h]

  if ( a4 == 1 )
  {
    sub_16A9F90(a1, a2, a3);
    return a1;
  }
  v22 = 1;
  v21 = 0;
  v24 = 1;
  v23 = 0;
  sub_16AE5C0(a2, a3, (__int64)&v21, (__int64)&v23);
  v6 = v24;
  if ( v24 <= 0x40 )
  {
    if ( v23 )
      goto LABEL_6;
LABEL_27:
    v19 = v22;
    v22 = 0;
    *(_DWORD *)(a1 + 8) = v19;
    *(_QWORD *)a1 = v21;
LABEL_28:
    if ( v6 <= 0x40 )
      goto LABEL_13;
LABEL_29:
    if ( v23 )
      j_j___libc_free_0_0(v23);
    goto LABEL_13;
  }
  if ( v6 - (unsigned int)sub_16A57B0((__int64)&v23) > 0x40 )
  {
    v7 = v6 - 1;
    v8 = 1LL << ((unsigned __int8)v6 - 1);
    if ( a4 )
      goto LABEL_33;
LABEL_18:
    v15 = v23[v7 >> 6];
    goto LABEL_19;
  }
  if ( !*v23 )
    goto LABEL_27;
LABEL_6:
  v7 = v6 - 1;
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( a4 )
  {
    if ( v6 <= 0x40 )
    {
      v9 = (unsigned __int64)v23;
LABEL_9:
      v10 = *(_DWORD *)(a3 + 8);
      v11 = (v9 & v8) == 0;
      v12 = *(_QWORD *)a3;
      if ( v10 > 0x40 )
        v12 = *(_QWORD *)(v12 + 8LL * ((v10 - 1) >> 6));
      v13 = v22;
      if ( ((v12 & (1LL << ((unsigned __int8)v10 - 1))) != 0) != !v11 )
        goto LABEL_12;
      v26 = v22;
      if ( v22 > 0x40 )
        sub_16A4FD0((__int64)&v25, &v21);
      else
        v25 = v21;
      sub_16A7490((__int64)&v25, 1);
      goto LABEL_25;
    }
LABEL_33:
    v9 = v23[v7 >> 6];
    goto LABEL_9;
  }
  if ( v6 > 0x40 )
    goto LABEL_18;
  v15 = (unsigned __int64)v23;
LABEL_19:
  v16 = *(_DWORD *)(a3 + 8);
  v17 = (v15 & v8) == 0;
  v18 = *(_QWORD *)a3;
  if ( v16 > 0x40 )
    v18 = *(_QWORD *)(v18 + 8LL * ((v16 - 1) >> 6));
  v13 = v22;
  if ( ((v18 & (1LL << ((unsigned __int8)v16 - 1))) != 0) != !v17 )
  {
    v26 = v22;
    if ( v22 > 0x40 )
      sub_16A4FD0((__int64)&v25, &v21);
    else
      v25 = v21;
    sub_16A7800((__int64)&v25, 1u);
LABEL_25:
    v6 = v24;
    *(_DWORD *)(a1 + 8) = v26;
    *(_QWORD *)a1 = v25;
    goto LABEL_28;
  }
LABEL_12:
  v14 = v21;
  *(_DWORD *)(a1 + 8) = v13;
  v22 = 0;
  *(_QWORD *)a1 = v14;
  if ( v6 > 0x40 )
    goto LABEL_29;
LABEL_13:
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
