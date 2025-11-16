// Function: sub_176EDE0
// Address: 0x176ede0
//
_QWORD *__fastcall sub_176EDE0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned int v5; // r15d
  int v6; // eax
  bool v7; // al
  int v8; // r13d
  bool v9; // cc
  int v10; // r15d
  _QWORD *v11; // r14
  unsigned int v12; // r15d
  unsigned int v13; // eax
  unsigned __int64 v14; // r8
  __int64 v15; // r8
  char v16; // cl
  unsigned int v17; // eax
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rax
  unsigned int v20; // ebx
  unsigned int v21; // r12d
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r12
  _QWORD *v25; // rax
  int v27; // eax
  char v28; // al
  bool v29; // al
  __int64 v30; // rcx
  bool v31; // al
  __int64 *v32; // rdi
  bool v33; // al
  __int64 v34; // rcx
  bool v35; // al
  __int64 v36; // rax
  __int64 v37; // r12
  _QWORD *v38; // rax
  unsigned int v39; // [rsp+8h] [rbp-B8h]
  __int64 v40; // [rsp+8h] [rbp-B8h]
  __int64 *v41; // [rsp+10h] [rbp-B0h]
  unsigned int v42; // [rsp+10h] [rbp-B0h]
  __int64 *v43; // [rsp+18h] [rbp-A8h]
  __int64 v44; // [rsp+18h] [rbp-A8h]
  __int64 v45; // [rsp+18h] [rbp-A8h]
  __int64 v46; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v47; // [rsp+28h] [rbp-98h]
  unsigned __int64 v48; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v49; // [rsp+38h] [rbp-88h]
  __int64 *v50; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v51; // [rsp+48h] [rbp-78h]
  __int16 v52; // [rsp+50h] [rbp-70h]
  __int64 **v53; // [rsp+60h] [rbp-60h] BYREF
  __int64 v54; // [rsp+68h] [rbp-58h]
  __int64 v55; // [rsp+70h] [rbp-50h] BYREF
  __int64 **v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h]

  v5 = *(_DWORD *)(a4 + 8);
  if ( v5 <= 0x40 )
  {
    v7 = *(_QWORD *)a4 == 1;
  }
  else
  {
    v43 = a3;
    v6 = sub_16A57B0(a4);
    a3 = v43;
    v7 = v5 - 1 == v6;
  }
  v8 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v9 = v5 <= 1;
  v44 = *(a3 - 3);
  v10 = v8;
  if ( !v9 && v7 && v8 == 40 )
  {
    v40 = (__int64)a3;
    v27 = sub_16431D0(*(_QWORD *)v44);
    a3 = (__int64 *)v40;
    if ( v27 )
    {
      v48 = 0;
      v53 = (__int64 **)&v48;
      v50 = 0;
      v54 = (unsigned int)(v27 - 1);
      v56 = &v50;
      v57 = v54;
      v28 = *(_BYTE *)(v44 + 16);
      if ( v28 == 51 )
      {
        v33 = sub_176AA50((__int64)&v53, *(_QWORD *)(v44 - 48), v40, (__int64)&v50);
        a3 = (__int64 *)v40;
        if ( v33 )
        {
          v35 = sub_176DE70((__int64)&v55, *(_QWORD *)(v44 - 24), v40, v34);
          a3 = (__int64 *)v40;
          if ( v35 )
          {
LABEL_50:
            if ( (__int64 *)v48 == v50 && v48 )
            {
              v45 = v48;
              v36 = sub_15A0680(*(_QWORD *)v48, 1, 0);
              LOWORD(v55) = 257;
              v37 = v36;
              v38 = sub_1648A60(56, 2u);
              v11 = v38;
              if ( v38 )
                sub_17582E0((__int64)v38, 40, v45, v37, (__int64)&v53);
              return v11;
            }
          }
        }
LABEL_51:
        v10 = *(_WORD *)(a2 + 18) & 0x7FFF;
        goto LABEL_5;
      }
      if ( v28 == 5 && *(_WORD *)(v44 + 18) == 27 )
      {
        v29 = sub_176ABC0(
                (__int64)&v53,
                *(_QWORD *)(v44 - 24LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF)),
                v40,
                4LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF));
        a3 = (__int64 *)v40;
        if ( v29 )
        {
          v31 = sub_176E260((__int64)&v53, v44, v40, v30);
          a3 = (__int64 *)v40;
          if ( v31 )
            goto LABEL_50;
        }
        goto LABEL_51;
      }
    }
  }
LABEL_5:
  v11 = 0;
  if ( (unsigned int)(v10 - 32) > 1 )
    return v11;
  v11 = (_QWORD *)a3[1];
  if ( !v11 )
    return v11;
  v11 = (_QWORD *)v11[1];
  if ( v11 )
    return 0;
  v39 = sub_16431D0(*a3);
  v12 = sub_16431D0(*(_QWORD *)v44);
  sub_14C2530((__int64)&v53, (__int64 *)v44, a1[333], 0, a1[330], a2, a1[332], 0);
  LOBYTE(v13) = v54;
  v49 = v54;
  if ( (unsigned int)v54 <= 0x40 )
  {
    v14 = (unsigned __int64)v53;
LABEL_10:
    v15 = v55 | v14;
LABEL_11:
    v16 = 64 - v13;
    v17 = 64;
    v18 = ~(v15 << v16);
    if ( v18 )
    {
      _BitScanReverse64(&v19, v18);
      v17 = v19 ^ 0x3F;
    }
    v20 = v12 - v39;
    goto LABEL_14;
  }
  sub_16A4FD0((__int64)&v48, (const void **)&v53);
  LOBYTE(v13) = v49;
  v20 = v12 - v39;
  if ( v49 <= 0x40 )
  {
    v14 = v48;
    goto LABEL_10;
  }
  sub_16A89F0((__int64 *)&v48, &v55);
  v13 = v49;
  v15 = v48;
  v49 = 0;
  v51 = v13;
  v50 = (__int64 *)v48;
  if ( v13 <= 0x40 )
    goto LABEL_11;
  v41 = (__int64 *)v48;
  v17 = sub_16A5810((__int64)&v50);
  if ( v41 )
  {
    v32 = v41;
    v42 = v17;
    j_j___libc_free_0_0(v32);
    v17 = v42;
    if ( v49 > 0x40 )
    {
      if ( v48 )
      {
        j_j___libc_free_0_0(v48);
        v17 = v42;
      }
    }
  }
LABEL_14:
  if ( v17 >= v20 )
  {
    sub_16A5C50((__int64)&v46, (const void **)a4, v12);
    v49 = v12;
    v21 = v39 - v12;
    if ( v12 > 0x40 )
    {
      sub_16A4EF0((__int64)&v48, 0, 0);
      v12 = v49;
      v39 = v49 + v21;
      if ( v49 == v49 + v21 )
        goto LABEL_59;
    }
    else
    {
      v48 = 0;
      if ( v39 == v12 )
      {
        v22 = 0;
        goto LABEL_20;
      }
    }
    if ( v39 <= 0x3F && v12 <= 0x40 )
    {
      v12 = v49;
      v22 = v48 | (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v21 + 64) << v39);
      goto LABEL_20;
    }
    sub_16A5260(&v48, v39, v12);
    v39 = v49;
LABEL_59:
    if ( v39 > 0x40 )
    {
      sub_16A8890((__int64 *)&v48, &v55);
      v12 = v49;
      v23 = v48;
      goto LABEL_21;
    }
    v22 = v48;
    v12 = v39;
LABEL_20:
    v23 = v55 & v22;
    v48 = v23;
LABEL_21:
    v51 = v12;
    v50 = (__int64 *)v23;
    v49 = 0;
    if ( v47 > 0x40 )
    {
      sub_16A89F0(&v46, (__int64 *)&v50);
      v12 = v51;
    }
    else
    {
      v46 |= v23;
    }
    if ( v12 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    v24 = sub_15A1070(*(_QWORD *)v44, (__int64)&v46);
    v52 = 257;
    v25 = sub_1648A60(56, 2u);
    v11 = v25;
    if ( v25 )
      sub_17582E0((__int64)v25, v8, v44, v24, (__int64)&v50);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
  }
  if ( (unsigned int)v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( (unsigned int)v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  return v11;
}
