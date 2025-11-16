// Function: sub_D96BA0
// Address: 0xd96ba0
//
__int64 __fastcall sub_D96BA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  unsigned int v6; // eax
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned int v12; // edx
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  unsigned int v16; // r8d
  bool v17; // cf
  const void *v18; // rcx
  const void *v19; // rax
  const void **v21; // rsi
  const void **v22; // rsi
  unsigned __int64 v23; // rcx
  const void *v24; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+8h] [rbp-58h]
  const void *v26; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-48h]
  const void *v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-38h]
  const void *v30; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-28h]

  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  v7 = *(_QWORD *)(v5 + 24);
  v8 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 > 0x40 )
  {
    v21 = (const void **)(v5 + 24);
    if ( (*(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6)) & v8) != 0 )
    {
      v31 = v6;
      sub_C43780((__int64)&v30, v21);
      v6 = v31;
      if ( v31 > 0x40 )
      {
        sub_C43D10((__int64)&v30);
LABEL_7:
        sub_C46250((__int64)&v30);
        v25 = v31;
        v24 = v30;
        goto LABEL_8;
      }
      v9 = (unsigned __int64)v30;
LABEL_4:
      v10 = ~v9 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v6);
      if ( !v6 )
        v10 = 0;
      v30 = (const void *)v10;
      goto LABEL_7;
    }
    v25 = v6;
    sub_C43780((__int64)&v24, v21);
  }
  else
  {
    v9 = *(_QWORD *)(v5 + 24);
    if ( (v8 & v7) != 0 )
    {
      v31 = *(_DWORD *)(v5 + 32);
      goto LABEL_4;
    }
    v25 = *(_DWORD *)(v5 + 32);
    v24 = (const void *)v7;
  }
LABEL_8:
  v11 = *(_QWORD *)(a3 + 32);
  v12 = *(_DWORD *)(v11 + 32);
  v13 = *(_QWORD *)(v11 + 24);
  v14 = 1LL << ((unsigned __int8)v12 - 1);
  if ( v12 > 0x40 )
  {
    v22 = (const void **)(v11 + 24);
    if ( (*(_QWORD *)(v13 + 8LL * ((v12 - 1) >> 6)) & v14) == 0 )
    {
      v27 = v12;
      sub_C43780((__int64)&v26, v22);
      v12 = v27;
LABEL_11:
      v16 = v25;
      v17 = v25 < v12;
      if ( v25 <= v12 )
        goto LABEL_12;
      goto LABEL_34;
    }
    v31 = v12;
    sub_C43780((__int64)&v30, v22);
    v12 = v31;
    if ( v31 > 0x40 )
    {
      sub_C43D10((__int64)&v30);
      goto LABEL_33;
    }
    v15 = (unsigned __int64)v30;
  }
  else
  {
    v15 = *(_QWORD *)(v11 + 24);
    if ( (v14 & v13) == 0 )
    {
      v27 = *(_DWORD *)(v11 + 32);
      v26 = (const void *)v13;
      goto LABEL_11;
    }
    v31 = *(_DWORD *)(v11 + 32);
  }
  v23 = ~v15 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
  if ( !v12 )
    v23 = 0;
  v30 = (const void *)v23;
LABEL_33:
  sub_C46250((__int64)&v30);
  v12 = v31;
  v16 = v25;
  v27 = v31;
  v26 = v30;
  v17 = v25 < v31;
  if ( v25 <= v31 )
  {
LABEL_12:
    if ( v17 )
    {
      sub_C449B0((__int64)&v30, &v24, v12);
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      v18 = v30;
      v16 = v31;
      v12 = v27;
      v19 = v26;
      v24 = v30;
    }
    else
    {
      v18 = v24;
      v19 = v26;
    }
    goto LABEL_14;
  }
LABEL_34:
  sub_C449B0((__int64)&v30, &v26, v16);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  v19 = v30;
  v12 = v31;
  v16 = v25;
  v18 = v24;
  v26 = v30;
LABEL_14:
  v31 = v12;
  v30 = v19;
  v27 = 0;
  v29 = v16;
  v28 = v18;
  v25 = 0;
  sub_C49E90(a1, (__int64)&v28, (__int64)&v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return a1;
}
