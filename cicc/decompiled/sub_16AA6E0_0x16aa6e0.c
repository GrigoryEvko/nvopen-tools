// Function: sub_16AA6E0
// Address: 0x16aa6e0
//
__int64 __fastcall sub_16AA6E0(__int64 a1, const void **a2)
{
  unsigned int v2; // ebx
  signed __int64 v3; // rax
  unsigned __int64 v4; // rcx
  unsigned int v5; // edx
  double v6; // xmm0_8
  double v7; // xmm0_8
  double v8; // xmm0_8
  unsigned __int64 v9; // rsi
  unsigned int v11; // eax
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  unsigned int v17; // ecx
  unsigned __int64 v18; // rdx
  unsigned int v19; // ebx
  unsigned int v20; // ecx
  unsigned int v21; // edx
  unsigned __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // eax
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned __int64 v28; // r12
  unsigned int v29; // [rsp+10h] [rbp-E0h]
  unsigned int v30; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v33; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v35; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v37; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-98h]
  __int64 v39; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-88h]
  unsigned __int64 v41; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+78h] [rbp-78h]
  unsigned __int64 v43; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+88h] [rbp-68h]
  unsigned __int64 v45; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v46; // [rsp+98h] [rbp-58h]
  unsigned __int64 v47; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v48; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v49; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v50; // [rsp+B8h] [rbp-38h]

  v2 = *((_DWORD *)a2 + 2);
  if ( v2 <= 0x40 )
  {
    v3 = (signed __int64)*a2;
    if ( !*a2 || (_BitScanReverse64(&v4, v3), v5 = 64 - (v4 ^ 0x3F), v5 <= 5) )
    {
      *(_DWORD *)(a1 + 8) = v2;
      *(_QWORD *)a1 = byte_42AEA40[v3] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
      return a1;
    }
    if ( v5 <= 0x33 )
    {
      if ( v3 >= 0 )
      {
LABEL_6:
        v6 = (double)(int)v3;
LABEL_7:
        if ( v6 < 0.0 )
        {
          v7 = sqrt(v6);
          v2 = *((_DWORD *)a2 + 2);
        }
        else
        {
          v7 = sqrt(v6);
        }
        v8 = round(v7);
        if ( v8 >= 9.223372036854776e18 )
          v9 = (unsigned int)(int)(v8 - 9.223372036854776e18) ^ 0x8000000000000000LL;
        else
          v9 = (unsigned int)(int)v8;
        *(_DWORD *)(a1 + 8) = v2;
        if ( v2 > 0x40 )
          sub_16A4EF0(a1, v9, 0);
        else
          *(_QWORD *)a1 = v9 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
        return a1;
      }
LABEL_19:
      v6 = (double)(int)(v3 & 1 | ((unsigned __int64)v3 >> 1)) + (double)(int)(v3 & 1 | ((unsigned __int64)v3 >> 1));
      goto LABEL_7;
    }
    v32 = *((_DWORD *)a2 + 2);
    v34 = v2;
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    v31 = v18 & 0x10;
    v13 = v2;
    goto LABEL_38;
  }
  v11 = v2 - sub_16A57B0((__int64)a2);
  if ( v11 <= 5 )
  {
    v12 = *(_QWORD *)*a2;
    *(_DWORD *)(a1 + 8) = v2;
    sub_16A4EF0(a1, byte_42AEA40[v12], 0);
    return a1;
  }
  if ( v11 <= 0x33 )
  {
    v3 = *(_QWORD *)*a2;
    if ( v3 >= 0 )
      goto LABEL_6;
    goto LABEL_19;
  }
  v32 = v2;
  sub_16A4EF0((__int64)&v31, 16, 0);
  v13 = *((_DWORD *)a2 + 2);
  v34 = v13;
  if ( v13 <= 0x40 )
  {
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
LABEL_38:
    v36 = v13;
    v33 = v18 & 1;
    goto LABEL_39;
  }
  sub_16A4EF0((__int64)&v33, 1, 0);
  v13 = *((_DWORD *)a2 + 2);
  v36 = v13;
  if ( v13 > 0x40 )
  {
    sub_16A4EF0((__int64)&v35, 0, 0);
    v14 = *((_DWORD *)a2 + 2);
    v38 = v14;
    if ( v14 <= 0x40 )
      v37 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & 2;
    else
      sub_16A4EF0((__int64)&v37, 2, 0);
    goto LABEL_24;
  }
  v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
LABEL_39:
  v38 = v13;
  v35 = 0;
  v37 = v18 & 2;
  if ( v2 > 4 )
  {
LABEL_24:
    v30 = 4;
    while ( 1 )
    {
      if ( (int)sub_16A9900((__int64)a2, &v31) <= 0 )
      {
LABEL_43:
        v19 = v30 >> 1;
        goto LABEL_44;
      }
      v17 = v32;
      v30 += 2;
      v50 = v32;
      if ( v32 <= 0x40 )
        break;
      sub_16A4FD0((__int64)&v49, (const void **)&v31);
      v17 = v50;
      if ( v50 <= 0x40 )
      {
        v15 = v32;
        goto LABEL_26;
      }
      sub_16A7DC0((__int64 *)&v49, 2u);
      v15 = v32;
LABEL_29:
      if ( v15 > 0x40 && v31 )
        j_j___libc_free_0_0(v31);
      v31 = v49;
      v32 = v50;
      if ( v30 >= v2 )
        goto LABEL_43;
    }
    v15 = v32;
    v49 = v31;
LABEL_26:
    v16 = 0;
    if ( v17 != 2 )
      v16 = (4 * v49) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
    v49 = v16;
    goto LABEL_29;
  }
  v19 = 2;
LABEL_44:
  v20 = v34;
  v50 = v34;
  if ( v34 <= 0x40 )
  {
    v21 = v34;
    v49 = v33;
LABEL_46:
    v22 = 0;
    if ( v20 != v19 )
      v22 = (v49 << v19) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v20);
    v49 = v22;
    goto LABEL_49;
  }
  sub_16A4FD0((__int64)&v49, (const void **)&v33);
  v20 = v50;
  if ( v50 <= 0x40 )
  {
    v21 = v34;
    goto LABEL_46;
  }
  sub_16A7DC0((__int64 *)&v49, v19);
  v21 = v34;
LABEL_49:
  if ( v21 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  v33 = v49;
  v34 = v50;
  while ( 1 )
  {
    sub_16A9D70((__int64)&v45, (__int64)a2, (__int64)&v33);
    sub_16A7200((__int64)&v45, (__int64 *)&v33);
    v23 = v46;
    v46 = 0;
    v48 = v23;
    v47 = v45;
    sub_16A9D70((__int64)&v49, (__int64)&v47, (__int64)&v37);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    v35 = v49;
    v36 = v50;
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
    if ( (int)sub_16A9900((__int64)&v33, &v35) <= 0 )
      break;
    if ( v34 <= 0x40 && v36 <= 0x40 )
    {
      v34 = v36;
      v33 = v35 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v36);
    }
    else
    {
      sub_16A51C0((__int64)&v33, (__int64)&v35);
    }
  }
  sub_16A7B50((__int64)&v39, (__int64)&v33, (__int64 *)&v33);
  v48 = v34;
  if ( v34 > 0x40 )
    sub_16A4FD0((__int64)&v47, (const void **)&v33);
  else
    v47 = v33;
  sub_16A7490((__int64)&v47, 1);
  v24 = v48;
  v48 = 0;
  v50 = v24;
  v49 = v47;
  v44 = v34;
  if ( v34 > 0x40 )
    sub_16A4FD0((__int64)&v43, (const void **)&v33);
  else
    v43 = v33;
  sub_16A7490((__int64)&v43, 1);
  v25 = v44;
  v44 = 0;
  v46 = v25;
  v45 = v43;
  sub_16A7B50((__int64)&v41, (__int64)&v45, (__int64 *)&v49);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( (int)sub_16A9900((__int64)a2, (unsigned __int64 *)&v39) >= 0 )
  {
    v48 = v42;
    if ( v42 > 0x40 )
      sub_16A4FD0((__int64)&v47, (const void **)&v41);
    else
      v47 = v41;
    sub_16A7590((__int64)&v47, &v39);
    v27 = v48;
    v48 = 0;
    v50 = v27;
    v49 = v47;
    sub_16A9D70((__int64)&v45, (__int64)&v49, (__int64)&v37);
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    v50 = *((_DWORD *)a2 + 2);
    if ( v50 > 0x40 )
      sub_16A4FD0((__int64)&v49, a2);
    else
      v49 = (unsigned __int64)*a2;
    sub_16A7590((__int64)&v49, &v39);
    v28 = v49;
    v29 = v50;
    v48 = v50;
    v47 = v49;
    if ( (int)sub_16A9900((__int64)&v47, &v45) >= 0 )
    {
      v50 = v34;
      if ( v34 > 0x40 )
        sub_16A4FD0((__int64)&v49, (const void **)&v33);
      else
        v49 = v33;
      sub_16A7490((__int64)&v49, 1);
      *(_DWORD *)(a1 + 8) = v50;
      *(_QWORD *)a1 = v49;
    }
    else
    {
      *(_DWORD *)(a1 + 8) = v34;
      v34 = 0;
      *(_QWORD *)a1 = v33;
    }
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
  }
  else
  {
    v26 = v34;
    v34 = 0;
    *(_DWORD *)(a1 + 8) = v26;
    *(_QWORD *)a1 = v33;
  }
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  return a1;
}
