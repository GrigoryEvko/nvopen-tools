// Function: sub_29B2390
// Address: 0x29b2390
//
__int64 __fastcall sub_29B2390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 **v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r15
  _BYTE *v15; // rsi
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int v20; // ecx
  __int64 *v21; // rdi
  __int64 v22; // r9
  const char *v23; // rax
  __int64 v24; // rcx
  __int64 *v25; // r15
  __int64 *v26; // r14
  _BYTE *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // edx
  __int64 *v32; // r8
  __int64 v33; // r10
  int v34; // r13d
  __int64 *v35; // rax
  const char *v36; // rax
  int v37; // esi
  const char *v38; // rax
  _BYTE *v39; // rsi
  __int64 *v40; // rax
  unsigned __int8 v41; // cl
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 *v45; // rbx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 i; // r15
  __int64 v49; // rax
  int v50; // edi
  const char *v51; // rax
  _BYTE *v52; // rsi
  int v53; // r8d
  const char *v54; // rax
  _BYTE *v55; // rsi
  unsigned __int8 *v56; // rbx
  __int64 *v57; // rcx
  __int64 v58; // r15
  __int64 *v59; // r12
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rsi
  unsigned int v63; // edx
  __int64 *v64; // rdi
  __int64 v65; // r9
  const char *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // rax
  __int64 v72; // r14
  __int64 *v73; // r13
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rsi
  unsigned int v78; // edx
  __int64 *v79; // r9
  __int64 v80; // r10
  const char *v81; // rax
  unsigned __int8 *v82; // rdi
  __int64 v83; // rdx
  __int64 *v84; // rdi
  const char *v85; // rax
  __int64 v86; // rdx
  int v88; // edi
  int v89; // ecx
  int v90; // r9d
  int v91; // ecx
  __int64 v92; // rdx
  __int64 v93; // rcx
  int v94; // ecx
  int v95; // r8d
  unsigned __int64 v99; // [rsp+20h] [rbp-E0h]
  __int64 v101; // [rsp+30h] [rbp-D0h]
  unsigned int v102; // [rsp+30h] [rbp-D0h]
  __int64 v103; // [rsp+38h] [rbp-C8h]
  __int64 v104; // [rsp+38h] [rbp-C8h]
  __int64 v105; // [rsp+40h] [rbp-C0h]
  __int64 *v106; // [rsp+40h] [rbp-C0h]
  __int64 v107; // [rsp+40h] [rbp-C0h]
  __int64 v108; // [rsp+58h] [rbp-A8h] BYREF
  const void *v109; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE *v110; // [rsp+68h] [rbp-98h]
  _BYTE *v111; // [rsp+70h] [rbp-90h]
  _QWORD *v112; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v113; // [rsp+88h] [rbp-78h]
  _BYTE *v114; // [rsp+90h] [rbp-70h]
  const char *v115; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v116; // [rsp+A8h] [rbp-58h]
  const char *v117; // [rsp+B0h] [rbp-50h]
  __int16 v118; // [rsp+C0h] [rbp-40h]

  v8 = a1;
  v9 = **(_QWORD **)(a1 + 88);
  v105 = *(_QWORD *)(v9 + 72);
  v10 = sub_AA4B30(v9);
  v11 = *(__int64 **)(a2 + 32);
  v109 = 0;
  v12 = (__int64 **)v10;
  v13 = *(unsigned int *)(a2 + 40);
  v110 = 0;
  v111 = 0;
  v14 = &v11[v13];
  v112 = 0;
  v113 = 0;
  v114 = 0;
  if ( v11 != v14 )
  {
    while ( 1 )
    {
      v16 = *v11;
      v17 = *(_BYTE *)(v8 + 8) == 0;
      v108 = *v11;
      if ( !v17 )
      {
        v18 = *(unsigned int *)(v8 + 232);
        v19 = *(_QWORD *)(v8 + 216);
        if ( !(_DWORD)v18 )
          goto LABEL_49;
        v20 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v21 = (__int64 *)(v19 + 8LL * v20);
        v22 = *v21;
        if ( v16 != *v21 )
        {
          v50 = 1;
          while ( v22 != -4096 )
          {
            v95 = v50 + 1;
            v20 = (v18 - 1) & (v50 + v20);
            v21 = (__int64 *)(v19 + 8LL * v20);
            v22 = *v21;
            if ( v16 == *v21 )
              goto LABEL_10;
            v50 = v95;
          }
LABEL_49:
          v51 = *(const char **)(v16 + 8);
          v52 = v113;
          v115 = v51;
          if ( v113 == v114 )
          {
            sub_9183A0((__int64)&v112, v113, &v115);
          }
          else
          {
            if ( v113 )
            {
              *(_QWORD *)v113 = v51;
              v52 = v113;
            }
            v113 = v52 + 8;
          }
          sub_29B2110(a6, &v108);
          goto LABEL_6;
        }
LABEL_10:
        if ( v21 == (__int64 *)(v19 + 8 * v18) )
          goto LABEL_49;
      }
      v23 = *(const char **)(v16 + 8);
      v15 = v110;
      v115 = v23;
      if ( v110 == v111 )
      {
        ++v11;
        sub_9183A0((__int64)&v109, v110, &v115);
        if ( v14 == v11 )
          break;
      }
      else
      {
        if ( v110 )
        {
          *(_QWORD *)v110 = v23;
          v15 = v110;
        }
        v110 = v15 + 8;
LABEL_6:
        if ( v14 == ++v11 )
          break;
      }
    }
  }
  v24 = *(_QWORD *)(a3 + 32);
  if ( v24 == v24 + 8LL * *(unsigned int *)(a3 + 40) )
    goto LABEL_26;
  v101 = a6;
  v25 = *(__int64 **)(a3 + 32);
  v26 = (__int64 *)(v24 + 8LL * *(unsigned int *)(a3 + 40));
  do
  {
    while ( 1 )
    {
      v28 = *v25;
      v17 = *(_BYTE *)(v8 + 8) == 0;
      v108 = *v25;
      if ( !v17 )
      {
        v29 = *(unsigned int *)(v8 + 232);
        v30 = *(_QWORD *)(v8 + 216);
        if ( !(_DWORD)v29 )
          goto LABEL_56;
        v31 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v32 = (__int64 *)(v30 + 8LL * v31);
        v33 = *v32;
        if ( v28 != *v32 )
        {
          v53 = 1;
          while ( v33 != -4096 )
          {
            v94 = v53 + 1;
            v31 = (v29 - 1) & (v53 + v31);
            v32 = (__int64 *)(v30 + 8LL * v31);
            v33 = *v32;
            if ( v28 == *v32 )
              goto LABEL_22;
            v53 = v94;
          }
LABEL_56:
          v54 = *(const char **)(v28 + 8);
          v55 = v113;
          v115 = v54;
          if ( v113 == v114 )
          {
            sub_9183A0((__int64)&v112, v113, &v115);
          }
          else
          {
            if ( v113 )
            {
              *(_QWORD *)v113 = v54;
              v55 = v113;
            }
            v113 = v55 + 8;
          }
          sub_29B2110(v101, &v108);
          goto LABEL_18;
        }
LABEL_22:
        if ( v32 == (__int64 *)(v30 + 8 * v29) )
          goto LABEL_56;
      }
      v34 = *((_DWORD *)v12 + 79);
      v35 = (__int64 *)sub_BD5C60(v28);
      v36 = (const char *)sub_BCE3C0(v35, v34);
      v27 = v110;
      v115 = v36;
      if ( v110 == v111 )
        break;
      if ( v110 )
      {
        *(_QWORD *)v110 = v36;
        v27 = v110;
      }
      v110 = v27 + 8;
LABEL_18:
      if ( v26 == ++v25 )
        goto LABEL_25;
    }
    ++v25;
    sub_9183A0((__int64)&v109, v110, &v115);
  }
  while ( v26 != v25 );
LABEL_25:
  a6 = v101;
LABEL_26:
  if ( v112 != (_QWORD *)v113 )
  {
    v37 = 0;
    *a7 = sub_BD0B90(*v12, v112, (v113 - (_BYTE *)v112) >> 3, 0);
    if ( !*(_BYTE *)(v8 + 200) )
      v37 = *((_DWORD *)v12 + 79);
    v38 = (const char *)sub_BCE3C0(*v12, v37);
    v39 = v110;
    v115 = v38;
    if ( v110 == v111 )
    {
      sub_9183A0((__int64)&v109, v110, &v115);
    }
    else
    {
      if ( v110 )
      {
        *(_QWORD *)v110 = v38;
        v39 = v110;
      }
      v110 = v39 + 8;
    }
  }
  v40 = (__int64 *)sub_29ABCE0(v8);
  v41 = 0;
  if ( *(_BYTE *)(v8 + 48) )
    v41 = *(_DWORD *)(*(_QWORD *)(v105 + 24) + 8LL) >> 8 != 0;
  v42 = (__int64)v109;
  v99 = sub_BCF480(v40, v109, (v110 - (_BYTE *)v109) >> 3, v41);
  v102 = *(_DWORD *)(*(_QWORD *)(v105 + 8) + 8LL) >> 8;
  v43 = sub_BD2DA0(136);
  v44 = v43;
  if ( v43 )
  {
    v42 = v99;
    sub_B2C3B0(v43, v99, 7, v102, a5, (__int64)v12);
  }
  if ( (*(_BYTE *)(v105 + 2) & 8) != 0 )
  {
    v42 = sub_B2E500(v105);
    sub_B2E8C0(v44, v42);
  }
  v108 = *(_QWORD *)(v105 + 120);
  v115 = (const char *)sub_A74680(&v108);
  v45 = (__int64 *)sub_A73280((__int64 *)&v115);
  for ( i = sub_A73290((__int64 *)&v115); (__int64 *)i != v45; ++v45 )
  {
    if ( !sub_A71840((__int64)v45) )
    {
      switch ( (unsigned int)sub_A71AE0(v45) )
      {
        case 0u:
        case 1u:
        case 2u:
        case 9u:
        case 0xEu:
        case 0xFu:
        case 0x15u:
        case 0x16u:
        case 0x1Cu:
        case 0x28u:
        case 0x2Bu:
        case 0x32u:
        case 0x33u:
        case 0x34u:
        case 0x36u:
        case 0x49u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x55u:
        case 0x56u:
        case 0x59u:
        case 0x5Au:
        case 0x5Bu:
        case 0x61u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
        case 0x65u:
          BUG();
        case 4u:
        case 6u:
        case 7u:
        case 8u:
        case 0x11u:
        case 0x14u:
        case 0x17u:
        case 0x1Au:
        case 0x20u:
        case 0x24u:
        case 0x27u:
        case 0x31u:
        case 0x35u:
        case 0x43u:
        case 0x4Cu:
        case 0x57u:
        case 0x58u:
        case 0x5Cu:
        case 0x5Du:
        case 0x5Eu:
          continue;
        default:
          goto LABEL_43;
      }
    }
    v49 = sub_A71FD0(v45);
    if ( v46 != 5 || *(_DWORD *)v49 != 1853188212 || *(_BYTE *)(v49 + 4) != 107 )
    {
LABEL_43:
      v42 = *v45;
      sub_B2CDC0(v44, *v45);
    }
  }
  if ( (*(_BYTE *)(v44 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v44, v42, v46, v47);
    if ( (*(_BYTE *)(v44 + 2) & 1) != 0 )
      sub_B2C6D0(v44, v42, v92, v93);
  }
  v56 = *(unsigned __int8 **)(v44 + 96);
  v57 = *(__int64 **)(a2 + 32);
  v106 = &v57[*(unsigned int *)(a2 + 40)];
  if ( v106 != v57 )
  {
    v103 = v8;
    v58 = a6;
    v59 = *(__int64 **)(a2 + 32);
    while ( 1 )
    {
      v60 = *(unsigned int *)(v58 + 24);
      v61 = *v59;
      v62 = *(_QWORD *)(v58 + 8);
      if ( !(_DWORD)v60 )
        goto LABEL_68;
      v63 = (v60 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v64 = (__int64 *)(v62 + 8LL * v63);
      v65 = *v64;
      if ( v61 != *v64 )
      {
        v88 = 1;
        while ( v65 != -4096 )
        {
          v89 = v88 + 1;
          v63 = (v60 - 1) & (v88 + v63);
          v64 = (__int64 *)(v62 + 8LL * v63);
          v65 = *v64;
          if ( v61 == *v64 )
            goto LABEL_67;
          v88 = v89;
        }
        goto LABEL_68;
      }
LABEL_67:
      if ( v64 != (__int64 *)(v62 + 8 * v60) )
      {
        if ( v106 == ++v59 )
          goto LABEL_73;
      }
      else
      {
LABEL_68:
        v66 = sub_BD5D20(*v59);
        v118 = 261;
        v115 = v66;
        v116 = v67;
        sub_BD6B50(v56, &v115);
        if ( (unsigned __int8)sub_BD6020(v61) )
        {
          if ( (*(_BYTE *)(v44 + 2) & 1) != 0 )
            sub_B2C6D0(v44, (__int64)&v115, v68, v69);
          sub_B2D3C0(v44, -858993459 * ((__int64)&v56[-*(_QWORD *)(v44 + 96)] >> 3), 74);
        }
        v56 += 40;
        if ( v106 == ++v59 )
        {
LABEL_73:
          v8 = v103;
          a6 = v58;
          break;
        }
      }
    }
  }
  v70 = *(_QWORD *)(a3 + 32);
  v71 = *(unsigned int *)(a3 + 40);
  if ( v70 + 8 * v71 != v70 )
  {
    v107 = v44;
    v72 = a6;
    v73 = (__int64 *)(v70 + 8 * v71);
    v104 = v8;
    v74 = *(__int64 **)(a3 + 32);
    while ( 1 )
    {
      v75 = *(unsigned int *)(v72 + 24);
      v76 = *v74;
      v77 = *(_QWORD *)(v72 + 8);
      if ( !(_DWORD)v75 )
        goto LABEL_80;
      v78 = (v75 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v79 = (__int64 *)(v77 + 8LL * v78);
      v80 = *v79;
      if ( v76 != *v79 )
      {
        v90 = 1;
        while ( v80 != -4096 )
        {
          v91 = v90 + 1;
          v78 = (v75 - 1) & (v90 + v78);
          v79 = (__int64 *)(v77 + 8LL * v78);
          v80 = *v79;
          if ( v76 == *v79 )
            goto LABEL_79;
          v90 = v91;
        }
        goto LABEL_80;
      }
LABEL_79:
      if ( v79 != (__int64 *)(v77 + 8 * v75) )
      {
        if ( v73 == ++v74 )
          goto LABEL_81;
      }
      else
      {
LABEL_80:
        v81 = sub_BD5D20(v76);
        ++v74;
        v82 = v56;
        v115 = v81;
        v56 += 40;
        v118 = 773;
        v116 = v83;
        v117 = ".out";
        sub_BD6B50(v82, &v115);
        if ( v73 == v74 )
        {
LABEL_81:
          v44 = v107;
          v8 = v104;
          break;
        }
      }
    }
  }
  v84 = *(__int64 **)(v8 + 16);
  if ( v84 )
  {
    v85 = (const char *)sub_FDC450(v84, a4);
    v116 = v86;
    v115 = v85;
    if ( (_BYTE)v86 )
      sub_B2F4C0(v44, (__int64)v115, 0, 0);
  }
  if ( v112 )
    j_j___libc_free_0((unsigned __int64)v112);
  if ( v109 )
    j_j___libc_free_0((unsigned __int64)v109);
  return v44;
}
