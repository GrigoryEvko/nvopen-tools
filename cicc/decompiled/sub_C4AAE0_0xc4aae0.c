// Function: sub_C4AAE0
// Address: 0xc4aae0
//
__int64 __fastcall sub_C4AAE0(__int64 a1, const void **a2)
{
  unsigned int v3; // ebx
  signed __int64 v4; // rax
  unsigned __int64 v5; // rcx
  unsigned int v6; // edx
  double v7; // xmm0_8
  double v8; // xmm0_8
  double v9; // xmm0_8
  _QWORD *v10; // rax
  unsigned __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // r14d
  unsigned int v18; // esi
  unsigned __int64 v19; // rcx
  unsigned int v20; // eax
  unsigned int v21; // r14d
  unsigned int v22; // eax
  unsigned int v23; // edi
  __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned int v28; // eax
  unsigned int v29; // eax
  unsigned int v30; // eax
  unsigned __int64 v31; // r12
  bool v32; // sf
  unsigned int v33; // eax
  unsigned int v34; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v36; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v37; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v38; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v39; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v40; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v41; // [rsp+58h] [rbp-A8h]
  __int64 v42; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-98h]
  __int64 v44; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-88h]
  unsigned __int64 v46; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v47; // [rsp+88h] [rbp-78h]
  unsigned __int64 v48; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+98h] [rbp-68h]
  unsigned __int64 v50; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v51; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v52; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+B8h] [rbp-48h]
  unsigned __int64 v54; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+C8h] [rbp-38h]

  v3 = *((_DWORD *)a2 + 2);
  if ( v3 <= 0x40 )
  {
    v4 = (signed __int64)*a2;
    if ( !*a2 || (_BitScanReverse64(&v5, v4), v6 = 64 - (v5 ^ 0x3F), v6 <= 5) )
    {
      v15 = byte_3F65880[v4];
      *(_DWORD *)(a1 + 8) = v3;
      *(_QWORD *)a1 = v15;
      return a1;
    }
    if ( v6 <= 0x33 )
    {
      if ( v4 >= 0 )
      {
LABEL_6:
        v7 = (double)(int)v4;
        goto LABEL_7;
      }
LABEL_19:
      v7 = (double)(int)(v4 & 1 | ((unsigned __int64)v4 >> 1)) + (double)(int)(v4 & 1 | ((unsigned __int64)v4 >> 1));
LABEL_7:
      if ( v7 < 0.0 )
      {
        v8 = sqrt(v7);
        v3 = *((_DWORD *)a2 + 2);
      }
      else
      {
        v8 = sqrt(v7);
      }
      v9 = round(v8);
      if ( v9 >= 9.223372036854776e18 )
      {
        v10 = (_QWORD *)a1;
        *(_DWORD *)(a1 + 8) = v3;
        v11 = (unsigned int)(int)(v9 - 9.223372036854776e18) ^ 0x8000000000000000LL;
        if ( v3 <= 0x40 )
          goto LABEL_11;
      }
      else
      {
        v10 = (_QWORD *)a1;
        v11 = (unsigned int)(int)v9;
        *(_DWORD *)(a1 + 8) = v3;
        if ( v3 <= 0x40 )
        {
LABEL_11:
          *v10 = v11;
          return a1;
        }
      }
      sub_C43690(a1, v11, 0);
      return a1;
    }
    v37 = *((_DWORD *)a2 + 2);
    v16 = v3;
    v36 = 16;
    v39 = v3;
LABEL_21:
    v41 = v16;
    v38 = 1;
    goto LABEL_22;
  }
  v12 = v3 - sub_C444A0((__int64)a2);
  if ( v12 <= 5 )
  {
    v13 = byte_3F65880[*(_QWORD *)*a2];
    *(_DWORD *)(a1 + 8) = v3;
    sub_C43690(a1, v13, 0);
    return a1;
  }
  if ( v12 <= 0x33 )
  {
    v4 = *(_QWORD *)*a2;
    if ( v4 >= 0 )
      goto LABEL_6;
    goto LABEL_19;
  }
  v37 = v3;
  sub_C43690((__int64)&v36, 16, 0);
  v16 = *((_DWORD *)a2 + 2);
  v39 = v16;
  if ( v16 <= 0x40 )
    goto LABEL_21;
  sub_C43690((__int64)&v38, 1, 0);
  v16 = *((_DWORD *)a2 + 2);
  v41 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43690((__int64)&v40, 0, 0);
    v43 = *((_DWORD *)a2 + 2);
    if ( v43 <= 0x40 )
      v42 = 2;
    else
      sub_C43690((__int64)&v42, 2, 0);
    goto LABEL_23;
  }
LABEL_22:
  v40 = 0;
  v43 = v16;
  v42 = 2;
  if ( v3 > 4 )
  {
LABEL_23:
    v17 = 4;
    while ( 1 )
    {
      if ( (int)sub_C49970((__int64)a2, &v36) <= 0 )
      {
LABEL_43:
        v21 = v17 >> 1;
        goto LABEL_44;
      }
      v20 = v37;
      v17 += 2;
      v55 = v37;
      if ( v37 <= 0x40 )
        break;
      sub_C43780((__int64)&v54, (const void **)&v36);
      v20 = v55;
      if ( v55 <= 0x40 )
      {
        v18 = v37;
LABEL_25:
        v19 = 0;
        if ( v20 != 2 && v20 )
          v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & (4 * v54);
        v54 = v19;
        goto LABEL_29;
      }
      sub_C47690((__int64 *)&v54, 2u);
      v18 = v37;
LABEL_29:
      if ( v18 > 0x40 && v36 )
        j_j___libc_free_0_0(v36);
      v36 = v54;
      v37 = v55;
      if ( v17 >= v3 )
        goto LABEL_43;
    }
    v18 = v37;
    v54 = v36;
    goto LABEL_25;
  }
  v21 = 2;
LABEL_44:
  v22 = v39;
  v55 = v39;
  if ( v39 <= 0x40 )
  {
    v23 = v39;
    v54 = v38;
LABEL_46:
    v24 = 0;
    if ( v21 != v22 )
      v24 = v54 << v21;
    v25 = v24 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
    if ( !v22 )
      v25 = 0;
    v54 = v25;
    goto LABEL_51;
  }
  sub_C43780((__int64)&v54, (const void **)&v38);
  v22 = v55;
  if ( v55 <= 0x40 )
  {
    v23 = v39;
    goto LABEL_46;
  }
  sub_C47690((__int64 *)&v54, v21);
  v23 = v39;
LABEL_51:
  if ( v23 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  v38 = v54;
  v39 = v55;
  while ( 1 )
  {
    sub_C4A1D0((__int64)&v50, (__int64)a2, (__int64)&v38);
    sub_C45EE0((__int64)&v50, (__int64 *)&v38);
    v26 = v51;
    v51 = 0;
    v53 = v26;
    v52 = v50;
    sub_C4A1D0((__int64)&v54, (__int64)&v52, (__int64)&v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    v40 = v54;
    v41 = v55;
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( (int)sub_C49970((__int64)&v38, &v40) <= 0 )
      break;
    if ( v39 <= 0x40 && v41 <= 0x40 )
    {
      v39 = v41;
      v38 = v40;
    }
    else
    {
      sub_C43990((__int64)&v38, (__int64)&v40);
    }
  }
  sub_C472A0((__int64)&v44, (__int64)&v38, (__int64 *)&v38);
  v53 = v39;
  if ( v39 > 0x40 )
    sub_C43780((__int64)&v52, (const void **)&v38);
  else
    v52 = v38;
  sub_C46A40((__int64)&v52, 1);
  v27 = v53;
  v53 = 0;
  v55 = v27;
  v54 = v52;
  v49 = v39;
  if ( v39 > 0x40 )
    sub_C43780((__int64)&v48, (const void **)&v38);
  else
    v48 = v38;
  sub_C46A40((__int64)&v48, 1);
  v28 = v49;
  v49 = 0;
  v51 = v28;
  v50 = v48;
  sub_C472A0((__int64)&v46, (__int64)&v50, (__int64 *)&v54);
  if ( v51 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
  if ( v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  if ( (int)sub_C49970((__int64)a2, (unsigned __int64 *)&v44) >= 0 )
  {
    v53 = v47;
    if ( v47 > 0x40 )
      sub_C43780((__int64)&v52, (const void **)&v46);
    else
      v52 = v46;
    sub_C46B40((__int64)&v52, &v44);
    v30 = v53;
    v53 = 0;
    v55 = v30;
    v54 = v52;
    sub_C4A1D0((__int64)&v50, (__int64)&v54, (__int64)&v42);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    v55 = *((_DWORD *)a2 + 2);
    if ( v55 > 0x40 )
      sub_C43780((__int64)&v54, a2);
    else
      v54 = (unsigned __int64)*a2;
    sub_C46B40((__int64)&v54, &v44);
    v31 = v54;
    v34 = v55;
    v53 = v55;
    v52 = v54;
    v32 = (int)sub_C49970((__int64)&v52, &v50) < 0;
    v33 = v39;
    if ( v32 )
    {
      v39 = 0;
      *(_DWORD *)(a1 + 8) = v33;
      *(_QWORD *)a1 = v38;
    }
    else
    {
      v55 = v39;
      if ( v39 > 0x40 )
        sub_C43780((__int64)&v54, (const void **)&v38);
      else
        v54 = v38;
      sub_C46A40((__int64)&v54, 1);
      *(_DWORD *)(a1 + 8) = v55;
      *(_QWORD *)a1 = v54;
    }
    if ( v34 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
  }
  else
  {
    v29 = v39;
    v39 = 0;
    *(_DWORD *)(a1 + 8) = v29;
    *(_QWORD *)a1 = v38;
  }
  if ( v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return a1;
}
