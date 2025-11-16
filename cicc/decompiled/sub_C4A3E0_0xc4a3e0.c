// Function: sub_C4A3E0
// Address: 0xc4a3e0
//
__int64 __fastcall sub_C4A3E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // edx
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // rbx
  unsigned int v8; // r9d
  unsigned __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // r8
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  unsigned int v19; // eax
  const void *v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned int v22; // eax
  unsigned int v23; // edx
  bool v24; // cc
  unsigned __int64 v26; // r8
  unsigned int v27; // eax
  const void *v28; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+8h] [rbp-68h]
  const void *v30; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-58h]
  unsigned __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-48h]
  unsigned __int64 v34; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  if ( v5 > 0x40 )
    v7 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
  v8 = *(_DWORD *)(a3 + 8);
  v9 = *(_QWORD *)a3;
  v10 = v8 - 1;
  v11 = 1LL << ((unsigned __int8)v8 - 1);
  v12 = (1LL << ((unsigned __int8)v5 - 1)) & v7;
  if ( v12 )
  {
    v13 = *(_QWORD *)a3;
    if ( v8 > 0x40 )
      v13 = *(_QWORD *)(v9 + 8LL * (v10 >> 6));
    v12 = v11 & v13;
    if ( v12 )
    {
      v29 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
      {
        sub_C43780((__int64)&v28, (const void **)a2);
        v5 = v29;
        if ( v29 > 0x40 )
        {
          sub_C43D10((__int64)&v28);
          goto LABEL_11;
        }
        v6 = (unsigned __int64)v28;
      }
      v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
      if ( !v5 )
        v14 = 0;
      v28 = (const void *)v14;
LABEL_11:
      sub_C46250((__int64)&v28);
      v15 = v29;
      v29 = 0;
      v31 = v15;
      v30 = v28;
      v16 = *(_DWORD *)(a3 + 8);
      v33 = v16;
      if ( v16 > 0x40 )
      {
        sub_C43780((__int64)&v32, (const void **)a3);
        v16 = v33;
        if ( v33 > 0x40 )
        {
          sub_C43D10((__int64)&v32);
          goto LABEL_16;
        }
        v17 = v32;
      }
      else
      {
        v17 = *(_QWORD *)a3;
      }
      v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17;
      if ( !v16 )
        v18 = 0;
      v32 = v18;
LABEL_16:
      sub_C46250((__int64)&v32);
      v19 = v33;
      v33 = 0;
      v35 = v19;
      v34 = v32;
      sub_C4A1D0(a1, (__int64)&v30, (__int64)&v34);
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
      if ( v29 <= 0x40 )
        return a1;
      v20 = v28;
      if ( !v28 )
        return a1;
LABEL_43:
      j_j___libc_free_0_0(v20);
      return a1;
    }
    v31 = *(_DWORD *)(a2 + 8);
    if ( v5 > 0x40 )
    {
      sub_C43780((__int64)&v30, (const void **)a2);
      v5 = v31;
      if ( v31 > 0x40 )
      {
        sub_C43D10((__int64)&v30);
        goto LABEL_51;
      }
      v6 = (unsigned __int64)v30;
    }
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
    if ( !v5 )
      v26 = 0;
    v30 = (const void *)v26;
LABEL_51:
    sub_C46250((__int64)&v30);
    v27 = v31;
    v31 = 0;
    v33 = v27;
    v32 = (unsigned __int64)v30;
    sub_C4A1D0((__int64)&v34, (__int64)&v32, a3);
    v23 = v35;
    if ( v35 > 0x40 )
      goto LABEL_52;
    goto LABEL_35;
  }
  if ( v8 <= 0x40 )
  {
    if ( (v11 & v9) != 0 )
    {
      v31 = *(_DWORD *)(a3 + 8);
      goto LABEL_31;
    }
LABEL_45:
    sub_C4A1D0(a1, a2, a3);
    return a1;
  }
  if ( (*(_QWORD *)(v9 + 8LL * (v10 >> 6)) & v11) == 0 )
    goto LABEL_45;
  v31 = *(_DWORD *)(a3 + 8);
  sub_C43780((__int64)&v30, (const void **)a3);
  v8 = v31;
  if ( v31 > 0x40 )
  {
    sub_C43D10((__int64)&v30);
    goto LABEL_34;
  }
  v9 = (unsigned __int64)v30;
LABEL_31:
  v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & ~v9;
  if ( !v8 )
    v21 = 0;
  v30 = (const void *)v21;
LABEL_34:
  sub_C46250((__int64)&v30);
  v22 = v31;
  v31 = 0;
  v33 = v22;
  v32 = (unsigned __int64)v30;
  sub_C4A1D0((__int64)&v34, a2, (__int64)&v32);
  v23 = v35;
  if ( v35 > 0x40 )
  {
LABEL_52:
    sub_C43D10((__int64)&v34);
    goto LABEL_38;
  }
LABEL_35:
  if ( v23 )
    v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & ~v34;
  v34 = v12;
LABEL_38:
  sub_C46250((__int64)&v34);
  v24 = v33 <= 0x40;
  *(_DWORD *)(a1 + 8) = v35;
  *(_QWORD *)a1 = v34;
  if ( !v24 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 )
  {
    v20 = v30;
    if ( v30 )
      goto LABEL_43;
  }
  return a1;
}
