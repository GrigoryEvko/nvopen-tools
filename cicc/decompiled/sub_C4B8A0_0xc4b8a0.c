// Function: sub_C4B8A0
// Address: 0xc4b8a0
//
__int64 __fastcall sub_C4B8A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r10
  unsigned int v9; // r9d
  unsigned __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // r8
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  unsigned int v21; // eax
  unsigned __int64 v22; // rax
  bool v23; // cc
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned int v26; // eax
  unsigned __int64 v28; // r8
  unsigned int v29; // eax
  unsigned __int64 v30; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-78h]
  unsigned __int64 v32; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-68h]
  unsigned __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-58h]
  unsigned __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-48h]
  unsigned __int64 v38; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v39; // [rsp+58h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 8);
  v7 = *(_QWORD *)a2;
  v8 = *(_QWORD *)a2;
  if ( v6 > 0x40 )
    v8 = *(_QWORD *)(v7 + 8LL * ((v6 - 1) >> 6));
  v9 = *(_DWORD *)(a3 + 8);
  v10 = *(_QWORD *)a3;
  v11 = v9 - 1;
  v12 = 1LL << ((unsigned __int8)v9 - 1);
  v13 = v8 & (1LL << ((unsigned __int8)v6 - 1));
  if ( !v13 )
  {
    if ( v9 > 0x40 )
    {
      if ( (*(_QWORD *)(v10 + 8LL * (v11 >> 6)) & v12) != 0 )
      {
        v37 = v9;
        sub_C43780((__int64)&v36, (const void **)a3);
        v9 = v37;
        v13 = 0;
        if ( v37 > 0x40 )
        {
          sub_C43D10((__int64)&v36);
          goto LABEL_38;
        }
        v10 = v36;
LABEL_35:
        v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & ~v10;
        if ( v9 )
          v13 = v25;
        v36 = v13;
LABEL_38:
        sub_C46250((__int64)&v36);
        v26 = v37;
        v37 = 0;
        v39 = v26;
        v38 = v36;
        sub_C4B490(a1, a2, (__int64)&v38);
        if ( v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
        if ( v37 > 0x40 )
        {
          v24 = v36;
          if ( v36 )
            goto LABEL_43;
        }
        return a1;
      }
    }
    else if ( (v12 & v10) != 0 )
    {
      v37 = v9;
      goto LABEL_35;
    }
    sub_C4B490(a1, a2, a3);
    return a1;
  }
  v14 = *(_QWORD *)a3;
  if ( v9 > 0x40 )
    v14 = *(_QWORD *)(v10 + 8LL * (v11 >> 6));
  v15 = v12 & v14;
  if ( v15 )
  {
    v31 = v6;
    if ( v6 > 0x40 )
    {
      sub_C43780((__int64)&v30, (const void **)a2);
      v6 = v31;
      if ( v31 > 0x40 )
      {
        sub_C43D10((__int64)&v30);
        goto LABEL_11;
      }
      v7 = v30;
    }
    v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v7;
    if ( !v6 )
      v16 = 0;
    v30 = v16;
LABEL_11:
    sub_C46250((__int64)&v30);
    v17 = v31;
    v31 = 0;
    v33 = v17;
    v32 = v30;
    v18 = *(_DWORD *)(a3 + 8);
    v35 = v18;
    if ( v18 > 0x40 )
    {
      sub_C43780((__int64)&v34, (const void **)a3);
      v18 = v35;
      if ( v35 > 0x40 )
      {
        sub_C43D10((__int64)&v34);
        goto LABEL_16;
      }
      v19 = v34;
    }
    else
    {
      v19 = *(_QWORD *)a3;
    }
    v20 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v19;
    if ( !v18 )
      v20 = 0;
    v34 = v20;
LABEL_16:
    sub_C46250((__int64)&v34);
    v21 = v35;
    v35 = 0;
    v37 = v21;
    v36 = v34;
    sub_C4B490((__int64)&v38, (__int64)&v32, (__int64)&v36);
    if ( v39 > 0x40 )
    {
      sub_C43D10((__int64)&v38);
    }
    else
    {
      v22 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v39) & ~v38;
      if ( !v39 )
        v22 = 0;
      v38 = v22;
    }
    sub_C46250((__int64)&v38);
    v23 = v37 <= 0x40;
    *(_DWORD *)(a1 + 8) = v39;
    *(_QWORD *)a1 = v38;
    if ( !v23 && v36 )
      j_j___libc_free_0_0(v36);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 <= 0x40 )
      return a1;
    v24 = v30;
    if ( !v30 )
      return a1;
LABEL_43:
    j_j___libc_free_0_0(v24);
    return a1;
  }
  v35 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43780((__int64)&v34, (const void **)a2);
    v6 = v35;
    if ( v35 > 0x40 )
    {
      sub_C43D10((__int64)&v34);
      goto LABEL_51;
    }
    v7 = v34;
  }
  v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v7;
  if ( !v6 )
    v28 = 0;
  v34 = v28;
LABEL_51:
  sub_C46250((__int64)&v34);
  v29 = v35;
  v35 = 0;
  v37 = v29;
  v36 = v34;
  sub_C4B490((__int64)&v38, (__int64)&v36, a3);
  if ( v39 > 0x40 )
  {
    sub_C43D10((__int64)&v38);
  }
  else
  {
    if ( v39 )
      v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v39) & ~v38;
    v38 = v15;
  }
  sub_C46250((__int64)&v38);
  v23 = v37 <= 0x40;
  *(_DWORD *)(a1 + 8) = v39;
  *(_QWORD *)a1 = v38;
  if ( !v23 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 )
  {
    v24 = v34;
    if ( v34 )
      goto LABEL_43;
  }
  return a1;
}
