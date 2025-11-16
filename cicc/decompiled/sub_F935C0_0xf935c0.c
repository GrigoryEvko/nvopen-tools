// Function: sub_F935C0
// Address: 0xf935c0
//
unsigned __int64 __fastcall sub_F935C0(
        __int64 a1,
        __int64 **a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned int *a5,
        _BYTE *a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v8; // r14
  unsigned __int64 result; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 *i; // rdx
  __int64 v15; // r12
  __int64 *v16; // r14
  unsigned int v17; // r13d
  __int64 v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rbx
  _QWORD *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r13
  _BYTE *v24; // rcx
  __int64 *v25; // rdi
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 **v28; // r13
  __int64 v29; // rax
  __int64 v30; // rbx
  _QWORD *v31; // r12
  char v32; // al
  unsigned int v33; // eax
  __int64 v34; // rbx
  unsigned int v35; // r14d
  unsigned int v36; // ecx
  __int64 v37; // rax
  bool v38; // al
  bool v39; // al
  __int64 v40; // r15
  unsigned int v41; // ecx
  unsigned __int8 *v42; // rax
  unsigned int v43; // edx
  unsigned __int8 *v44; // rsi
  unsigned __int64 v45; // r15
  unsigned int v46; // esi
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  const void **v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rax
  char v57; // bl
  char v58; // al
  __int64 v61; // [rsp+18h] [rbp-2F8h]
  __int64 v63; // [rsp+30h] [rbp-2E0h]
  __int64 v64; // [rsp+30h] [rbp-2E0h]
  const char *v66; // [rsp+48h] [rbp-2C8h]
  char v67; // [rsp+48h] [rbp-2C8h]
  char v68; // [rsp+5Fh] [rbp-2B1h] BYREF
  __int64 v69; // [rsp+60h] [rbp-2B0h] BYREF
  unsigned int v70; // [rsp+68h] [rbp-2A8h]
  __int64 v71; // [rsp+70h] [rbp-2A0h] BYREF
  unsigned int v72; // [rsp+78h] [rbp-298h]
  __int64 v73; // [rsp+80h] [rbp-290h] BYREF
  unsigned int v74; // [rsp+88h] [rbp-288h]
  unsigned __int64 v75; // [rsp+90h] [rbp-280h] BYREF
  unsigned int v76; // [rsp+98h] [rbp-278h]
  __int64 v77; // [rsp+A0h] [rbp-270h] BYREF
  unsigned int v78; // [rsp+A8h] [rbp-268h]
  __int64 v79; // [rsp+B0h] [rbp-260h]
  __int64 v80; // [rsp+B8h] [rbp-258h]
  __int16 v81; // [rsp+C0h] [rbp-250h]
  __int64 *v82; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v83; // [rsp+D8h] [rbp-238h]
  _QWORD v84[70]; // [rsp+E0h] [rbp-230h] BYREF

  v8 = a1;
  result = *(_QWORD *)a5;
  v11 = a3;
  v12 = *(_QWORD *)(*(_QWORD *)a5 + 8LL);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 8) = v12;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  v61 = *(_QWORD *)(v12 + 8);
  v82 = v84;
  v83 = 0x4000000000LL;
  if ( !v11 )
  {
    v63 = a5[2];
    if ( !a5[2] )
    {
LABEL_31:
      *(_DWORD *)v8 = 0;
      goto LABEL_32;
    }
    goto LABEL_9;
  }
  v13 = v84;
  if ( (unsigned __int64)v11 > 0x40 )
  {
    sub_C8D5F0((__int64)&v82, v84, v11, 8u, (__int64)a5, (__int64)a6);
    v13 = &v82[(unsigned int)v83];
    for ( i = &v82[a3]; i != v13; ++v13 )
    {
LABEL_4:
      if ( v13 )
        *v13 = 0;
    }
  }
  else
  {
    i = &v84[a3];
    if ( i != v84 )
      goto LABEL_4;
  }
  LODWORD(v83) = a3;
  v63 = a5[2];
  if ( !a5[2] )
  {
    v24 = *(_BYTE **)(a1 + 8);
    goto LABEL_36;
  }
  result = *(_QWORD *)a5;
LABEL_9:
  v15 = 0;
  v16 = (__int64 *)(a4 + 24);
  while ( 1 )
  {
    v21 = (_QWORD *)(16 * v15 + result);
    v22 = *v21;
    v20 = (_BYTE *)v21[1];
    v78 = *(_DWORD *)(*v21 + 32LL);
    if ( v78 <= 0x40 )
      v77 = *(_QWORD *)(v22 + 24);
    else
      sub_C43780((__int64)&v77, (const void **)(v22 + 24));
    v11 = (__int64)v16;
    sub_C46B40((__int64)&v77, v16);
    v17 = v78;
    v78 = 0;
    v76 = v17;
    v18 = v77;
    v75 = v77;
    v66 = (const char *)v77;
    if ( v17 > 0x40 )
    {
      if ( v17 - (unsigned int)sub_C444A0((__int64)&v75) <= 0x40 )
      {
        v23 = *(_QWORD *)v66;
      }
      else
      {
        if ( !v66 )
        {
          v18 = 0x1FFFFFFFFFFFFFFFLL;
          goto LABEL_15;
        }
        v23 = -1;
      }
      j_j___libc_free_0_0(v66);
      v18 = v23;
      if ( v78 > 0x40 && v77 )
      {
        j_j___libc_free_0_0(v77);
        v18 = v23;
      }
    }
LABEL_15:
    v82[v18] = (__int64)v20;
    v19 = *(_BYTE **)(a1 + 8);
    if ( v19 && *v20 != 13 && v20 != v19 )
    {
      if ( *v19 != 13 )
        v20 = 0;
      *(_QWORD *)(a1 + 8) = v20;
    }
    if ( ++v15 == v63 )
      break;
    result = *(_QWORD *)a5;
  }
  v8 = a1;
  v24 = *(_BYTE **)(a1 + 8);
  result = a5[2];
  if ( a3 <= result )
    goto LABEL_30;
LABEL_36:
  v11 = a3;
  v26 = 0;
  do
  {
    while ( 1 )
    {
      v27 = &v82[v26];
      if ( !*v27 )
        break;
      if ( a3 == ++v26 )
        goto LABEL_40;
    }
    ++v26;
    *v27 = (__int64)a6;
  }
  while ( a3 != v26 );
LABEL_40:
  result = (unsigned __int64)a6;
  if ( *a6 == 13 || a6 == v24 )
  {
LABEL_30:
    if ( !v24 )
      goto LABEL_43;
    goto LABEL_31;
  }
  *(_QWORD *)(v8 + 8) = 0;
LABEL_43:
  if ( *(_BYTE *)(v61 + 8) != 12 )
    goto LABEL_44;
  v70 = 1;
  v69 = 0;
  v72 = 1;
  v71 = 0;
  if ( !a3 )
  {
    v67 = 0;
LABEL_121:
    *(_QWORD *)(v8 + 32) = *v82;
    v52 = sub_ACCFD0(*a2, (__int64)&v71);
    *(_QWORD *)(v8 + 40) = v52;
    v53 = *(_DWORD *)(v52 + 32);
    v74 = v53;
    if ( v53 > 0x40 )
    {
      sub_C43780((__int64)&v73, (const void **)(v52 + 24));
      v53 = v74;
      v68 = 1;
      v11 = a3 - 1;
      if ( v74 > 0x3F )
      {
        v78 = v74;
        if ( v74 != 64 )
        {
          sub_C43690((__int64)&v77, v11, 0);
          goto LABEL_128;
        }
LABEL_127:
        v77 = v11;
LABEL_128:
        v11 = (__int64)&v73;
        sub_C4A7C0((__int64)&v75, (__int64)&v73, (__int64)&v77, (bool *)&v68);
        sub_969240((__int64 *)&v75);
        sub_969240(&v77);
        goto LABEL_129;
      }
    }
    else
    {
      v54 = *(_QWORD *)(v52 + 24);
      v68 = 1;
      v73 = v54;
      v11 = a3 - 1;
      if ( v53 == 64 )
      {
        v78 = 64;
        goto LABEL_127;
      }
    }
    if ( v53 )
    {
      v55 = 1LL << ((unsigned __int8)v53 - 1);
      v56 = v55 - 1;
      if ( v11 < -v55 )
        goto LABEL_129;
    }
    else
    {
      if ( v11 < 0 )
        goto LABEL_129;
      v56 = 0;
    }
    if ( v11 <= v56 )
    {
      v78 = v53;
      goto LABEL_127;
    }
LABEL_129:
    v57 = v67;
    v58 = v68;
    *(_DWORD *)v8 = 1;
    if ( !v67 )
      v57 = v58;
    *(_BYTE *)(v8 + 48) = v57;
    sub_969240(&v73);
    sub_969240(&v71);
    result = sub_969240(&v69);
    goto LABEL_32;
  }
  v67 = 0;
  v34 = 0;
  v64 = v8;
  while ( 1 )
  {
    v40 = v82[v34];
    if ( *(_BYTE *)v40 != 17 )
    {
      if ( *(_BYTE *)v40 != 13 || (v40 = *(_QWORD *)(*(_QWORD *)a5 + 8LL), *(_BYTE *)v40 != 17) )
      {
        v8 = v64;
        goto LABEL_76;
      }
    }
    if ( v34 )
      break;
LABEL_68:
    if ( v70 <= 0x40 && *(_DWORD *)(v40 + 32) <= 0x40u )
    {
      v50 = *(_QWORD *)(v40 + 24);
      v70 = *(_DWORD *)(v40 + 32);
      v69 = v50;
    }
    else
    {
      sub_C43990((__int64)&v69, v40 + 24);
    }
    if ( a3 == ++v34 )
    {
      v8 = v64;
      goto LABEL_121;
    }
  }
  v78 = *(_DWORD *)(v40 + 32);
  if ( v78 > 0x40 )
    sub_C43780((__int64)&v77, (const void **)(v40 + 24));
  else
    v77 = *(_QWORD *)(v40 + 24);
  sub_C46B40((__int64)&v77, &v69);
  v35 = v78;
  v76 = v78;
  v75 = v77;
  if ( v34 == 1 )
  {
    if ( v72 <= 0x40 && v78 <= 0x40 )
    {
      v71 = v77;
      v72 = v78;
      goto LABEL_108;
    }
    sub_C43990((__int64)&v71, (__int64)&v75);
    v35 = v76;
    v36 = v76 - 1;
    v37 = 1LL << ((unsigned __int8)v76 - 1);
    if ( v76 > 0x40 )
      goto LABEL_61;
LABEL_109:
    if ( (v37 & v75) != 0 )
      goto LABEL_104;
    v38 = v75 == 0;
LABEL_63:
    if ( v38 )
LABEL_104:
      v39 = (int)sub_C4C880(v40 + 24, (__int64)&v69) > 0;
    else
      v39 = (int)sub_C4C880(v40 + 24, (__int64)&v69) <= 0;
    v67 |= v39;
    if ( v35 > 0x40 && v75 )
      j_j___libc_free_0_0(v75);
    goto LABEL_68;
  }
  if ( v78 > 0x40 )
  {
    if ( !sub_C43C50((__int64)&v75, (const void **)&v71) )
    {
      v8 = v64;
      goto LABEL_136;
    }
    v36 = v35 - 1;
    v37 = 1LL << ((unsigned __int8)v35 - 1);
LABEL_61:
    if ( (*(_QWORD *)(v75 + 8LL * (v36 >> 6)) & v37) == 0 )
    {
      v38 = v35 == (unsigned int)sub_C444A0((__int64)&v75);
      goto LABEL_63;
    }
    goto LABEL_104;
  }
  if ( v77 == v71 )
  {
LABEL_108:
    v37 = 1LL << ((unsigned __int8)v78 - 1);
    goto LABEL_109;
  }
  v8 = v64;
LABEL_136:
  sub_969240((__int64 *)&v75);
LABEL_76:
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( *(_BYTE *)(v61 + 8) == 12 )
  {
    v41 = *(_DWORD *)(v61 + 8) >> 8;
    if ( a3 < 0xFFFFFFFF / v41 )
    {
      v42 = *(unsigned __int8 **)(a7 + 32);
      v43 = a3 * v41;
      v44 = &v42[*(_QWORD *)(a7 + 40)];
      if ( v42 != v44 )
      {
        while ( v43 > *v42 )
        {
          if ( v44 == ++v42 )
            goto LABEL_44;
        }
        v76 = a3 * v41;
        if ( v43 > 0x40 )
          sub_C43690((__int64)&v75, 0, 0);
        else
          v75 = 0;
        v45 = a3;
        do
        {
          v46 = *(_DWORD *)(v61 + 8) >> 8;
          if ( v76 > 0x40 )
          {
            sub_C47690((__int64 *)&v75, v46);
          }
          else
          {
            v47 = 0;
            if ( v46 != v76 )
              v47 = v75 << v46;
            v48 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v76) & v47;
            if ( !v76 )
              v48 = 0;
            v75 = v48;
          }
          v49 = (const void **)v82[--v45];
          if ( (unsigned int)*(unsigned __int8 *)v49 - 12 > 1 )
          {
            sub_C449B0((__int64)&v77, v49 + 3, v76);
            if ( v76 > 0x40 )
              sub_C43BD0(&v75, &v77);
            else
              v75 |= v77;
            if ( v78 > 0x40 && v77 )
              j_j___libc_free_0_0(v77);
          }
        }
        while ( v45 );
        v11 = (__int64)&v75;
        v51 = sub_ACCFD0(*a2, (__int64)&v75);
        *(_DWORD *)v8 = 2;
        *(_QWORD *)(v8 + 16) = v51;
        *(_QWORD *)(v8 + 24) = v61;
        result = sub_969240((__int64 *)&v75);
LABEL_32:
        v25 = v82;
        if ( v82 == v84 )
          return result;
        return _libc_free(v25, v11);
      }
    }
  }
LABEL_44:
  v28 = (__int64 **)sub_BCD420((__int64 *)v61, a3);
  v29 = sub_AD1300(v28, v82, (unsigned int)v83);
  BYTE4(v75) = 0;
  v30 = v29;
  v81 = 1283;
  v77 = (__int64)"switch.table.";
  v79 = *a8;
  v80 = a8[1];
  v31 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v31 )
    sub_B30000((__int64)v31, (__int64)a2, v28, 1, 8, v30, (__int64)&v77, 0, 0, v75, 0);
  v32 = *((_BYTE *)v31 + 32);
  *(_QWORD *)(v8 + 56) = v31;
  *((_BYTE *)v31 + 32) = v32 & 0x3F | 0x80;
  v33 = sub_AE5260(a7, v61);
  v11 = v33;
  result = sub_B2F770((__int64)v31, v33);
  *(_DWORD *)v8 = 3;
  v25 = v82;
  if ( v82 != v84 )
    return _libc_free(v25, v11);
  return result;
}
