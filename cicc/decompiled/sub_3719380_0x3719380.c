// Function: sub_3719380
// Address: 0x3719380
//
__int64 __fastcall sub_3719380(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // ebx
  __int64 v4; // r13
  unsigned int v5; // eax
  unsigned int v6; // r14d
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rcx
  bool v10; // zf
  unsigned __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // eax
  unsigned int v16; // edx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  unsigned int v20; // eax
  unsigned __int64 v21; // rcx
  int v22; // eax
  bool v23; // al
  unsigned int v24; // ecx
  int v25; // eax
  unsigned int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rax
  bool v29; // cc
  unsigned int v31; // edx
  unsigned __int64 v32; // rax
  unsigned int v33; // [rsp+14h] [rbp-DCh]
  unsigned int v34; // [rsp+20h] [rbp-D0h]
  unsigned int v35; // [rsp+20h] [rbp-D0h]
  int v36; // [rsp+24h] [rbp-CCh]
  const void *v37; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v39; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-A8h]
  const void *v41; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-98h]
  unsigned __int64 v43; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-88h]
  unsigned __int64 v45; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v46; // [rsp+78h] [rbp-78h]
  unsigned __int64 v47; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+88h] [rbp-68h]
  unsigned __int64 v49; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+98h] [rbp-58h]
  unsigned __int64 v51; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v52; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v53; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v54; // [rsp+B8h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 8);
  v3 = v2 - 1;
  v40 = v2;
  v38 = 1;
  v37 = 0;
  v4 = 1LL << ((unsigned __int8)v2 - 1);
  if ( v2 <= 0x40 )
  {
    v39 = 0;
LABEL_3:
    v39 |= v4;
    goto LABEL_4;
  }
  sub_C43690((__int64)&v39, 0, 0);
  if ( v40 <= 0x40 )
    goto LABEL_3;
  *(_QWORD *)(v39 + 8LL * (v3 >> 6)) |= v4;
LABEL_4:
  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  v5 = *(_DWORD *)(a2 + 8);
  v6 = v5 - 1;
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 <= 0x40 )
  {
    v8 = *(_QWORD *)a2;
    if ( (*(_QWORD *)a2 & v7) == 0 )
    {
      v54 = *(_DWORD *)(a2 + 8);
      v42 = v5;
      v41 = (const void *)v8;
      goto LABEL_95;
    }
    v54 = *(_DWORD *)(a2 + 8);
    goto LABEL_7;
  }
  if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * (v6 >> 6)) & v7) != 0 )
  {
    v54 = *(_DWORD *)(a2 + 8);
    sub_C43780((__int64)&v53, (const void **)a2);
    v5 = v54;
    if ( v54 > 0x40 )
    {
      sub_C43D10((__int64)&v53);
LABEL_10:
      sub_C46250((__int64)&v53);
      v42 = v54;
      v41 = (const void *)v53;
      goto LABEL_11;
    }
    v8 = v53;
LABEL_7:
    v9 = ~v8 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
    v10 = v5 == 0;
    v11 = 0;
    if ( !v10 )
      v11 = v9;
    v53 = v11;
    goto LABEL_10;
  }
  v42 = *(_DWORD *)(a2 + 8);
  sub_C43780((__int64)&v41, (const void **)a2);
LABEL_11:
  v5 = *(_DWORD *)(a2 + 8);
  v54 = v5;
  v6 = v5 - 1;
  if ( v5 <= 0x40 )
  {
LABEL_95:
    v53 = *(_QWORD *)a2;
    goto LABEL_96;
  }
  sub_C43780((__int64)&v53, (const void **)a2);
  v5 = v54;
  if ( v54 > 0x40 )
  {
    sub_C482E0((__int64)&v53, v6);
    goto LABEL_14;
  }
LABEL_96:
  if ( v5 == v6 )
    v53 = 0;
  else
    v53 >>= v6;
LABEL_14:
  sub_C45EE0((__int64)&v53, (__int64 *)&v39);
  v44 = v54;
  v43 = v53;
  sub_C4B490((__int64)&v53, (__int64)&v43, (__int64)&v41);
  v50 = v44;
  if ( v44 > 0x40 )
    sub_C43780((__int64)&v49, (const void **)&v43);
  else
    v49 = v43;
  sub_C46F20((__int64)&v49, 1u);
  v12 = v50;
  v50 = 0;
  v52 = v12;
  v51 = v49;
  if ( v54 > 0x40 )
  {
    sub_C43D10((__int64)&v53);
  }
  else
  {
    v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v54) & ~v53;
    if ( !v54 )
      v13 = 0;
    v53 = v13;
  }
  sub_C46250((__int64)&v53);
  sub_C45EE0((__int64)&v53, (__int64 *)&v51);
  v14 = v54;
  v54 = 0;
  v46 = v14;
  v45 = v53;
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  v48 = 1;
  v15 = *(_DWORD *)(a2 + 8);
  v47 = 0;
  v50 = 1;
  v49 = 0;
  v36 = v15 - 1;
  v52 = 1;
  v51 = 0;
  v54 = 1;
  v53 = 0;
  sub_C4BFE0((__int64)&v39, (__int64)&v45, &v47, &v49);
  sub_C4BFE0((__int64)&v39, (__int64)&v41, &v51, &v53);
  v16 = v48;
  do
  {
    while ( 1 )
    {
      do
      {
        ++v36;
        if ( v16 > 0x40 )
        {
          sub_C47690((__int64 *)&v47, 1u);
        }
        else
        {
          v17 = 0;
          if ( v16 >= 2 )
            v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & (2 * v47);
          v47 = v17;
        }
        if ( v50 > 0x40 )
        {
          sub_C47690((__int64 *)&v49, 1u);
        }
        else
        {
          v18 = 0;
          if ( v50 >= 2 )
            v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v50) & (2 * v49);
          v49 = v18;
        }
        if ( (int)sub_C49970((__int64)&v49, &v45) >= 0 )
        {
          sub_C46250((__int64)&v47);
          sub_C46B40((__int64)&v49, (__int64 *)&v45);
        }
        if ( v52 > 0x40 )
        {
          sub_C47690((__int64 *)&v51, 1u);
          v20 = v54;
          if ( v54 > 0x40 )
            goto LABEL_101;
        }
        else
        {
          v19 = 0;
          if ( v52 >= 2 )
            v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v52) & (2 * v51);
          v20 = v54;
          v51 = v19;
          if ( v54 > 0x40 )
          {
LABEL_101:
            sub_C47690((__int64 *)&v53, 1u);
            goto LABEL_47;
          }
        }
        v21 = 0;
        if ( v20 >= 2 )
          v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & (2 * v53);
        v53 = v21;
LABEL_47:
        if ( (int)sub_C49970((__int64)&v53, (unsigned __int64 *)&v41) >= 0 )
        {
          sub_C46250((__int64)&v51);
          sub_C46B40((__int64)&v53, (__int64 *)&v41);
        }
        if ( v38 <= 0x40 && v42 <= 0x40 )
        {
          v38 = v42;
          v37 = v41;
        }
        else
        {
          sub_C43990((__int64)&v37, (__int64)&v41);
        }
        sub_C46B40((__int64)&v37, (__int64 *)&v53);
        v22 = sub_C49970((__int64)&v47, (unsigned __int64 *)&v37);
        v16 = v48;
      }
      while ( v22 < 0 );
      if ( v48 > 0x40 )
        break;
      if ( (const void *)v47 != v37 )
        goto LABEL_57;
      v24 = v50;
      if ( v50 <= 0x40 )
        goto LABEL_107;
LABEL_56:
      v33 = v16;
      v35 = v24;
      v25 = sub_C444A0((__int64)&v49);
      v16 = v33;
      if ( v35 != v25 )
        goto LABEL_57;
    }
    v34 = v48;
    v23 = sub_C43C50((__int64)&v47, &v37);
    v16 = v34;
    if ( !v23 )
      break;
    v24 = v50;
    if ( v50 > 0x40 )
      goto LABEL_56;
LABEL_107:
    ;
  }
  while ( !v49 );
LABEL_57:
  if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
    j_j___libc_free_0_0(*(_QWORD *)a1);
  *(_QWORD *)a1 = v51;
  v26 = v52;
  v52 = 0;
  *(_DWORD *)(a1 + 8) = v26;
  sub_C46250(a1);
  v27 = *(_DWORD *)(a2 + 8);
  v28 = 1LL << ((unsigned __int8)v27 - 1);
  if ( v27 > 0x40 )
  {
    if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v27 - 1) >> 6)) & v28) == 0 )
      goto LABEL_62;
  }
  else if ( (*(_QWORD *)a2 & v28) == 0 )
  {
    goto LABEL_62;
  }
  v31 = *(_DWORD *)(a1 + 8);
  if ( v31 <= 0x40 )
  {
    v32 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v31) & ~*(_QWORD *)a1;
    if ( !v31 )
      v32 = 0;
    *(_QWORD *)a1 = v32;
  }
  else
  {
    sub_C43D10(a1);
  }
  sub_C46250(a1);
  v27 = *(_DWORD *)(a2 + 8);
LABEL_62:
  v29 = v54 <= 0x40;
  *(_DWORD *)(a1 + 16) = v36 - v27;
  if ( !v29 && v53 )
    j_j___libc_free_0_0(v53);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0((unsigned __int64)v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0((unsigned __int64)v37);
  return a1;
}
