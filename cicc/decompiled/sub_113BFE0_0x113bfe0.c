// Function: sub_113BFE0
// Address: 0x113bfe0
//
_QWORD *__fastcall sub_113BFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // r13
  __int16 v10; // di
  unsigned int v11; // eax
  __int64 v12; // rcx
  unsigned int v13; // r8d
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // r13
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // r12
  _QWORD *v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  unsigned int v37; // edx
  __int64 v38; // r8
  bool v39; // al
  bool v40; // bl
  unsigned __int64 v41; // rax
  unsigned int v42; // eax
  bool v43; // bl
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  unsigned int v47; // eax
  __int64 v48; // r12
  unsigned int v49; // ebx
  char v50; // al
  unsigned int v51; // eax
  __int64 v52; // rdi
  unsigned int v53; // ecx
  int v54; // eax
  unsigned int v55; // eax
  __int16 v56; // ax
  unsigned int v57; // edx
  __int64 v58; // rax
  __int64 v59; // r12
  _QWORD *v60; // rax
  unsigned int v61; // ebx
  __int64 v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  const void *v65; // r9
  bool v66; // al
  __int64 v67; // rdi
  unsigned int v68; // eax
  bool v69; // r12
  _QWORD *v70; // rax
  int v71; // eax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  unsigned int v75; // eax
  unsigned int v76; // eax
  _QWORD *v77; // rax
  const void *v78; // [rsp+0h] [rbp-C0h]
  char v80; // [rsp+8h] [rbp-B8h]
  __int16 v81; // [rsp+10h] [rbp-B0h]
  __int64 v82; // [rsp+10h] [rbp-B0h]
  const void *v83; // [rsp+10h] [rbp-B0h]
  __int64 v84; // [rsp+10h] [rbp-B0h]
  char v85; // [rsp+10h] [rbp-B0h]
  __int64 v86; // [rsp+18h] [rbp-A8h]
  unsigned int v88; // [rsp+18h] [rbp-A8h]
  char v89; // [rsp+18h] [rbp-A8h]
  unsigned int v90; // [rsp+18h] [rbp-A8h]
  __int16 v91; // [rsp+18h] [rbp-A8h]
  char v92; // [rsp+18h] [rbp-A8h]
  char v93; // [rsp+18h] [rbp-A8h]
  char v94; // [rsp+18h] [rbp-A8h]
  unsigned int v95; // [rsp+18h] [rbp-A8h]
  bool v96; // [rsp+18h] [rbp-A8h]
  char v97; // [rsp+2Fh] [rbp-91h] BYREF
  __int64 v98; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v99; // [rsp+38h] [rbp-88h]
  __int64 v100; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v101; // [rsp+48h] [rbp-78h]
  __int64 v102; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v103; // [rsp+58h] [rbp-68h]
  const void *v104; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v105; // [rsp+68h] [rbp-58h]
  __int16 v106; // [rsp+80h] [rbp-40h]

  v4 = a3;
  v7 = *(_QWORD *)(a3 - 32);
  v8 = *(_QWORD *)(a3 - 64);
  v9 = v7 + 24;
  if ( *(_BYTE *)v7 != 17 )
  {
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v25 = sub_AD7630(v7, 0, v24);
    if ( !v25 || *v25 != 17 )
      return 0;
    v4 = a3;
    v9 = (__int64)(v25 + 24);
  }
  v10 = *(_WORD *)(a2 + 2);
  v86 = v4;
  v97 = 0;
  v81 = v10 & 0x3F;
  LOBYTE(v11) = sub_9893F0(v10 & 0x3F, a4, &v97);
  v14 = v11;
  if ( (_BYTE)v11 )
  {
    v18 = *(unsigned int *)(v9 + 8);
    v19 = 1LL << ((unsigned __int8)v18 - 1);
    v20 = *(_QWORD *)v9;
    if ( (unsigned int)v18 > 0x40 )
    {
      if ( (*(_QWORD *)(v20 + 8LL * ((unsigned int)(v18 - 1) >> 6)) & v19) != 0 )
      {
LABEL_10:
        v21 = *(_QWORD *)(v8 + 8);
        if ( v97 )
        {
          v22 = sub_AD62B0(v21);
          v106 = 257;
          v23 = sub_BD2C40(72, unk_3F10FD0);
          v16 = v23;
          if ( v23 )
            sub_1113300((__int64)v23, 38, v8, v22, (__int64)&v104);
        }
        else
        {
          v32 = sub_AD6530(v21, v18);
          v106 = 257;
          v33 = sub_BD2C40(72, unk_3F10FD0);
          v16 = v33;
          if ( v33 )
            sub_1113300((__int64)v33, 40, v8, v32, (__int64)&v104);
        }
        return v16;
      }
    }
    else if ( (v20 & v19) != 0 )
    {
      goto LABEL_10;
    }
    v16 = (_QWORD *)a2;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v26 = *(_QWORD *)(a2 - 8);
    else
      v26 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v27 = *(_QWORD **)v26;
    if ( *(_QWORD *)v26 )
    {
      v28 = *(_QWORD *)(v26 + 8);
      **(_QWORD **)(v26 + 16) = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = *(_QWORD *)(v26 + 16);
    }
    *(_QWORD *)v26 = v8;
    if ( v8 )
    {
      v29 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v26 + 8) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = v26 + 8;
      *(_QWORD *)(v26 + 16) = v8 + 16;
      *(_QWORD *)(v8 + 16) = v26;
    }
    if ( *(_BYTE *)v27 > 0x1Cu )
    {
      v104 = v27;
      v30 = *(_QWORD *)(a1 + 40) + 2096LL;
      sub_1134860(v30, (__int64 *)&v104);
      v31 = v27[2];
      if ( v31 )
      {
        if ( !*(_QWORD *)(v31 + 8) )
        {
          v104 = *(const void **)(v31 + 24);
          sub_1134860(v30, (__int64 *)&v104);
        }
      }
    }
    return v16;
  }
  v15 = *(_QWORD *)(v86 + 16);
  if ( v15 )
  {
    if ( !*(_QWORD *)(v15 + 8) )
    {
      v49 = *(_WORD *)(a2 + 2) & 0x3F;
      if ( v49 - 32 > 1 )
      {
        v89 = v14;
        v50 = sub_986B30((__int64 *)v9, a4, v14, v12, v13);
        LOBYTE(v14) = v89;
        if ( v50 )
        {
          v56 = sub_B53550(v49);
          goto LABEL_67;
        }
        v51 = *(_DWORD *)(v9 + 8);
        v52 = *(_QWORD *)v9;
        v53 = v51 - 1;
        if ( v51 <= 0x40 )
        {
          if ( v52 == (1LL << v53) - 1 )
          {
LABEL_66:
            v55 = sub_B53550(v49);
            v56 = sub_B52F50(v55);
LABEL_67:
            v91 = v56;
            sub_9865C0((__int64)&v100, a4);
            v57 = v101;
            if ( v101 > 0x40 )
            {
              sub_C43C10(&v100, (__int64 *)v9);
              v57 = v101;
              v58 = v100;
            }
            else
            {
              v58 = *(_QWORD *)v9 ^ v100;
              v100 = v58;
            }
            v103 = v57;
            v102 = v58;
            v101 = 0;
            v59 = sub_AD8D80(*(_QWORD *)(v8 + 8), (__int64)&v102);
            v106 = 257;
            v60 = sub_BD2C40(72, unk_3F10FD0);
            v16 = v60;
            if ( v60 )
              sub_1113300((__int64)v60, v91, v8, v59, (__int64)&v104);
            goto LABEL_71;
          }
        }
        else
        {
          v90 = v51 - 1;
          if ( (*(_QWORD *)(v52 + 8LL * (v53 >> 6)) & (1LL << v53)) == 0 )
          {
            v80 = v14;
            v54 = sub_C445E0(v9);
            LOBYTE(v14) = v80;
            if ( v54 == v90 )
              goto LABEL_66;
          }
        }
      }
    }
  }
  if ( v81 != 34 )
  {
    if ( v81 != 36 )
      return 0;
    v34 = *(_DWORD *)(a4 + 8);
    v103 = v34;
    if ( v34 > 0x40 )
    {
      sub_C43780((__int64)&v102, (const void **)a4);
      v34 = v103;
      if ( v103 > 0x40 )
      {
        sub_C43D10((__int64)&v102);
        goto LABEL_39;
      }
      v35 = v102;
    }
    else
    {
      v35 = *(_QWORD *)a4;
    }
    v36 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & ~v35;
    if ( !v34 )
      v36 = 0;
    v102 = v36;
LABEL_39:
    sub_C46250((__int64)&v102);
    v37 = v103;
    v38 = v102;
    v103 = 0;
    v105 = v37;
    v104 = (const void *)v102;
    if ( *(_DWORD *)(v9 + 8) <= 0x40u )
    {
      v40 = 0;
      if ( *(_QWORD *)v9 != v102 )
        goto LABEL_41;
    }
    else
    {
      v82 = v102;
      v88 = v37;
      v39 = sub_C43C50(v9, &v104);
      v37 = v88;
      v38 = v82;
      v40 = v39;
      if ( !v39 )
        goto LABEL_41;
    }
    if ( *(_DWORD *)(a4 + 8) > 0x40u )
    {
      v84 = v38;
      v95 = v37;
      v71 = sub_C44630(a4);
      v38 = v84;
      v37 = v95;
      v40 = v71 == 1;
    }
    else
    {
      v40 = 0;
      if ( *(_QWORD *)a4 )
        v40 = (*(_QWORD *)a4 & (*(_QWORD *)a4 - 1LL)) == 0;
    }
LABEL_41:
    if ( v37 > 0x40 )
    {
      if ( v38 )
      {
        j_j___libc_free_0_0(v38);
        if ( v103 > 0x40 )
        {
          if ( v102 )
            j_j___libc_free_0_0(v102);
        }
      }
    }
    if ( v40 )
    {
      sub_9865C0((__int64)&v100, a4);
      sub_987160((__int64)&v100, a4, v72, v73, v74);
      v75 = v101;
      v101 = 0;
      v103 = v75;
      v102 = v100;
      v48 = sub_AD8D80(*(_QWORD *)(v8 + 8), (__int64)&v102);
      v106 = 257;
      v16 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v16 )
        goto LABEL_71;
    }
    else
    {
      if ( *(_DWORD *)(v9 + 8) <= 0x40u )
      {
        if ( *(_QWORD *)v9 != *(_QWORD *)a4 )
          return 0;
      }
      else if ( !sub_C43C50(v9, (const void **)a4) )
      {
        return 0;
      }
      sub_9865C0((__int64)&v102, a4);
      if ( v103 > 0x40 )
      {
        sub_C43D10((__int64)&v102);
      }
      else
      {
        v41 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v103) & ~v102;
        if ( !v103 )
          v41 = 0;
        v102 = v41;
      }
      sub_C46250((__int64)&v102);
      v42 = v103;
      v103 = 0;
      v105 = v42;
      v104 = (const void *)v102;
      v43 = sub_986BA0((__int64)&v104);
      sub_969240((__int64 *)&v104);
      sub_969240(&v102);
      if ( !v43 )
        return 0;
      sub_9865C0((__int64)&v100, a4);
      sub_987160((__int64)&v100, a4, v44, v45, v46);
      v47 = v101;
      v101 = 0;
      v103 = v47;
      v102 = v100;
      v48 = sub_AD8D80(*(_QWORD *)(v8 + 8), (__int64)&v102);
      v106 = 257;
      v16 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v16 )
      {
LABEL_71:
        sub_969240(&v102);
        sub_969240(&v100);
        return v16;
      }
    }
    sub_1113300((__int64)v16, 34, v8, v48, (__int64)&v104);
    goto LABEL_71;
  }
  v61 = *(_DWORD *)(a4 + 8);
  v103 = v61;
  if ( v61 > 0x40 )
  {
    v85 = v14;
    sub_C43780((__int64)&v102, (const void **)a4);
    v61 = v103;
    LOBYTE(v14) = v85;
    if ( v103 > 0x40 )
    {
      sub_C43D10((__int64)&v102);
      v61 = v103;
      v65 = (const void *)v102;
      LOBYTE(v14) = v85;
      goto LABEL_77;
    }
    v62 = v102;
  }
  else
  {
    v62 = *(_QWORD *)a4;
  }
  v63 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v61) & ~v62;
  v64 = 0;
  if ( v61 )
    v64 = v63;
  v102 = v64;
  v65 = (const void *)v64;
LABEL_77:
  v105 = v61;
  v104 = v65;
  v103 = 0;
  if ( *(_DWORD *)(v9 + 8) <= 0x40u )
  {
    if ( v65 != *(const void **)v9 )
      goto LABEL_79;
LABEL_101:
    v78 = v65;
    sub_9865C0((__int64)&v98, a4);
    sub_C46A40((__int64)&v98, 1);
    v76 = v99;
    v99 = 0;
    v101 = v76;
    v100 = v98;
    v96 = sub_986BA0((__int64)&v100);
    sub_969240(&v100);
    sub_969240(&v98);
    v65 = v78;
    LOBYTE(v14) = v96;
    goto LABEL_79;
  }
  v83 = v65;
  v92 = v14;
  v66 = sub_C43C50(v9, &v104);
  LOBYTE(v14) = v92;
  v65 = v83;
  if ( v66 )
    goto LABEL_101;
LABEL_79:
  if ( v61 > 0x40 && v65 )
  {
    v93 = v14;
    j_j___libc_free_0_0(v65);
    LOBYTE(v14) = v93;
  }
  if ( v103 > 0x40 && v102 )
  {
    v94 = v14;
    j_j___libc_free_0_0(v102);
    LOBYTE(v14) = v94;
  }
  if ( (_BYTE)v14 )
  {
    v106 = 257;
    v77 = sub_BD2C40(72, unk_3F10FD0);
    v16 = v77;
    if ( v77 )
      sub_1113300((__int64)v77, 36, v8, v7, (__int64)&v104);
  }
  else
  {
    v67 = v9;
    v16 = 0;
    if ( !sub_AAD8B0(v67, (_QWORD *)a4) )
      return v16;
    sub_9865C0((__int64)&v102, a4);
    sub_C46A40((__int64)&v102, 1);
    v68 = v103;
    v103 = 0;
    v105 = v68;
    v104 = (const void *)v102;
    v69 = sub_986BA0((__int64)&v104);
    sub_969240((__int64 *)&v104);
    sub_969240(&v102);
    if ( !v69 )
      return 0;
    v106 = 257;
    v70 = sub_BD2C40(72, unk_3F10FD0);
    v16 = v70;
    if ( v70 )
      sub_1113300((__int64)v70, 34, v8, v7, (__int64)&v104);
  }
  return v16;
}
