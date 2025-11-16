// Function: sub_3068370
// Address: 0x3068370
//
unsigned __int64 __fastcall sub_3068370(
        __int64 a1,
        int a2,
        _QWORD **a3,
        unsigned int a4,
        unsigned int *a5,
        signed __int64 a6,
        unsigned __int8 a7,
        unsigned int a8,
        int a9,
        char a10,
        char a11)
{
  unsigned __int64 v11; // r13
  int v14; // r13d
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r13
  _QWORD *v21; // rbx
  __int64 v22; // r12
  unsigned int v23; // r13d
  __int16 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // r14d
  unsigned __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // r9
  __int64 v31; // r8
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned int *i; // r9
  unsigned int v35; // ecx
  unsigned int v36; // edx
  __int64 v37; // r11
  unsigned int v38; // esi
  __int64 v39; // rax
  int v40; // edx
  __int64 v41; // rcx
  unsigned __int64 v42; // rbx
  char v43; // r8
  char v44; // cl
  signed __int64 v45; // rax
  bool v46; // of
  unsigned __int64 v47; // rbx
  int v48; // edx
  unsigned int v49; // r11d
  unsigned int v50; // r14d
  unsigned int v51; // ebx
  _QWORD *v52; // r9
  unsigned int *v53; // r14
  unsigned int v54; // esi
  unsigned int v55; // edi
  unsigned int v56; // eax
  unsigned int v57; // ebx
  __int64 *v58; // r14
  __int64 *v59; // r12
  __int64 v60; // rdi
  __int64 v61; // rax
  int v62; // edx
  __int64 v63; // rdx
  __int64 *v64; // rax
  unsigned __int64 *v65; // r10
  __int64 *v66; // r14
  __int64 v67; // r15
  unsigned __int64 v68; // r13
  signed __int64 v69; // rax
  __int64 v70; // r13
  int v71; // eax
  __int64 v72; // rax
  size_t v73; // rdx
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rax
  __int64 v77; // [rsp+0h] [rbp-120h]
  __int64 v78; // [rsp+8h] [rbp-118h]
  _QWORD *v79; // [rsp+10h] [rbp-110h]
  unsigned int v80; // [rsp+10h] [rbp-110h]
  __int64 v81; // [rsp+18h] [rbp-108h]
  unsigned int s; // [rsp+20h] [rbp-100h]
  void *sa; // [rsp+20h] [rbp-100h]
  __int16 v84; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v85; // [rsp+28h] [rbp-F8h]
  __int64 v86; // [rsp+28h] [rbp-F8h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  unsigned int v88; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v89; // [rsp+30h] [rbp-F0h]
  __int64 v93; // [rsp+58h] [rbp-C8h]
  __int64 *v95; // [rsp+60h] [rbp-C0h]
  unsigned int v96; // [rsp+68h] [rbp-B8h]
  int v97; // [rsp+6Ch] [rbp-B4h]
  unsigned __int64 v98; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v99; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v100; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v101; // [rsp+88h] [rbp-98h]
  unsigned __int64 v102; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v103; // [rsp+98h] [rbp-88h]
  void *v104; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v105; // [rsp+A8h] [rbp-78h]
  _QWORD v106[6]; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v107; // [rsp+E0h] [rbp-40h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v96 = *((_DWORD *)a3 + 8);
  v14 = v96 / a4;
  v81 = sub_BCDA70(a3[3], v96 / a4);
  if ( a10 || a11 )
  {
    v89 = sub_3067E40(a1, a2, a3, a7, 1, 0, a9, 0);
    v97 = v48;
  }
  else
  {
    v15 = a7;
    BYTE1(v15) = 1;
    v89 = sub_30670E0(a1, a2, (__int64)a3, v15, a8, a9);
    v97 = v16;
  }
  v21 = *a3;
  v17 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  s = v14;
  v19 = v18;
  v20 = (__int64)v21;
  for ( LOWORD(v21) = v17; ; LOWORD(v21) = v105 )
  {
    LOWORD(v17) = (_WORD)v21;
    sub_2FE6CC0((__int64)&v104, *(_QWORD *)(a1 + 24), v20, v17, v19);
    if ( (_BYTE)v104 == 10 )
      break;
    if ( !(_BYTE)v104 )
      goto LABEL_12;
    if ( (_WORD)v105 == (_WORD)v21 )
    {
      if ( (_WORD)v21 )
      {
LABEL_12:
        LODWORD(v21) = (unsigned __int16)v21;
        v22 = a1;
        v23 = s;
        v24 = (__int16)v21;
        goto LABEL_13;
      }
      if ( v106[0] == v19 )
      {
        v24 = (__int16)v21;
        v22 = a1;
        v23 = s;
        LODWORD(v21) = 0;
        goto LABEL_13;
      }
    }
    v17 = v105;
    v19 = v106[0];
  }
  v22 = a1;
  v23 = s;
  LODWORD(v21) = (unsigned __int16)v21;
  v24 = (__int16)v21;
  if ( !(_WORD)v21 )
  {
    LODWORD(v21) = 8;
    v24 = 8;
  }
LABEL_13:
  v84 = v24;
  v25 = sub_9208B0(*(_QWORD *)(v22 + 8), (__int64)a3);
  v105 = v26;
  v104 = (void *)((unsigned __int64)(v25 + 7) >> 3);
  v27 = sub_CA1930(&v104);
  if ( (unsigned __int16)v84 <= 1u || (unsigned __int16)(v84 - 504) <= 7u )
    BUG();
  v28 = *(_QWORD *)&byte_444C4A0[16 * (int)v21 - 16] + 7LL;
  LOBYTE(v105) = byte_444C4A0[16 * (int)v21 - 8];
  v104 = (void *)(v28 >> 3);
  v29 = sub_CA1930(&v104);
  v31 = (__int64)&a5[a6];
  if ( !v97 && v27 > v29 )
  {
    v49 = (v27 - (v27 != 0)) / v29 + (v27 != 0);
    v50 = (v49 + 63) >> 6;
    v51 = (v96 - (v96 != 0)) / v49 + (v96 != 0);
    v104 = v106;
    v105 = 0x600000000LL;
    if ( v50 > 6 )
    {
      v80 = v49;
      sub_C8D5F0((__int64)&v104, v106, v50, 8u, v31, v30);
      memset(v104, 0, 8LL * v50);
      LODWORD(v105) = v50;
      v52 = v104;
      v49 = v80;
      v31 = (__int64)&a5[a6];
    }
    else
    {
      if ( v50 )
      {
        v73 = 8LL * v50;
        if ( v73 )
        {
          v88 = v49;
          memset(v106, 0, v73);
          v31 = (__int64)&a5[a6];
          v49 = v88;
        }
      }
      LODWORD(v105) = v50;
      v52 = v106;
    }
    v107 = v49;
    if ( a5 != (unsigned int *)v31 )
    {
      v53 = a5;
      do
      {
        v54 = *v53;
        if ( a4 <= v96 )
        {
          v55 = 0;
          while ( 1 )
          {
            ++v55;
            v56 = v54 / v51;
            v54 += a4;
            v52[v56 >> 6] |= 1LL << v56;
            if ( v23 <= v55 )
              break;
            v52 = v104;
          }
          v52 = v104;
        }
        ++v53;
      }
      while ( (unsigned int *)v31 != v53 );
    }
    v85 = v49;
    if ( &v52[(unsigned int)v105] == v52 )
    {
      v89 = 0;
    }
    else
    {
      v79 = v52;
      v57 = 0;
      v58 = &v52[(unsigned int)v105];
      v78 = v31;
      v77 = v22;
      v59 = v52;
      do
      {
        v60 = *v59++;
        v57 += sub_39FAC40(v60);
      }
      while ( v59 != v58 );
      v52 = v79;
      v31 = v78;
      v22 = v77;
      v89 = (v89 * v57 != 0) + (v89 * v57 - (v89 * v57 != 0)) / v85;
    }
    if ( v52 != v106 )
    {
      v86 = v31;
      _libc_free((unsigned __int64)v52);
      v31 = v86;
    }
  }
  v99 = v23;
  if ( v23 > 0x40 )
  {
    v87 = v31;
    sub_C43690((__int64)&v98, -1, 1);
    v31 = v87;
  }
  else
  {
    v32 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v23;
    if ( a4 > v96 )
      v32 = 0;
    v98 = v32;
  }
  v101 = v96;
  if ( v96 > 0x40 )
  {
    sa = (void *)v31;
    sub_C43690((__int64)&v100, -1, 1);
    v103 = v96;
    sub_C43690((__int64)&v102, 0, 0);
    v31 = (__int64)sa;
  }
  else
  {
    v103 = v96;
    v102 = 0;
    v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v96;
    if ( !v96 )
      v33 = 0;
    v100 = v33;
  }
  for ( i = a5; i != (unsigned int *)v31; ++i )
  {
    v35 = *i;
    v36 = 0;
    if ( a4 <= v96 )
    {
      do
      {
        while ( 1 )
        {
          v37 = 1LL << v35;
          if ( v103 > 0x40 )
            break;
          ++v36;
          v35 += a4;
          v102 |= v37;
          if ( v23 <= v36 )
            goto LABEL_31;
        }
        v38 = v35;
        ++v36;
        v35 += a4;
        *(_QWORD *)(v102 + 8LL * (v38 >> 6)) |= v37;
      }
      while ( v23 > v36 );
    }
LABEL_31:
    ;
  }
  if ( a2 != 32 )
  {
    v39 = sub_3064F80(v22, v81, (__int64 *)&v98, 0, 1);
    v41 = v39 * a6;
    if ( is_mul_ok(v39, a6) )
    {
      v42 = v41 + v89;
      if ( __OFADD__(v41, v89) )
      {
        v42 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v41 <= 0 )
          v42 = 0x8000000000000000LL;
      }
      goto LABEL_35;
    }
    if ( v39 <= 0 )
    {
      if ( v39 < 0 && a6 < 0 )
        goto LABEL_124;
    }
    else if ( a6 > 0 )
    {
LABEL_124:
      if ( v40 != 1 )
      {
        v42 = 0x7FFFFFFFFFFFFFFFLL;
        v75 = v89 + 0x7FFFFFFFFFFFFFFFLL;
        if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v89) )
        {
LABEL_35:
          v43 = 0;
          v44 = 1;
          goto LABEL_36;
        }
LABEL_126:
        v42 = v75;
        goto LABEL_35;
      }
      v76 = v89 + 0x7FFFFFFFFFFFFFFFLL;
      if ( __OFADD__(v89, 0x7FFFFFFFFFFFFFFFLL) )
      {
        v42 = 0x7FFFFFFFFFFFFFFFLL;
        goto LABEL_35;
      }
      goto LABEL_133;
    }
    v42 = 0x8000000000000000LL;
    if ( v40 != 1 )
    {
      v75 = v89 + 0x8000000000000000LL;
      if ( __OFADD__(0x8000000000000000LL, v89) )
        goto LABEL_35;
      goto LABEL_126;
    }
    v76 = v89 + 0x8000000000000000LL;
    if ( __OFADD__(0x8000000000000000LL, v89) )
      goto LABEL_35;
LABEL_133:
    v42 = v76;
    goto LABEL_35;
  }
  v61 = sub_3064F80(v22, v81, (__int64 *)&v98, 1, 0);
  if ( v62 == 1 )
  {
    if ( is_mul_ok(v61, a6) )
    {
      v63 = v61 * a6;
      goto LABEL_69;
    }
    if ( a6 <= 0 )
    {
      if ( v61 < 0 && a6 < 0 )
      {
        v42 = v89 + 0x7FFFFFFFFFFFFFFFLL;
        if ( !__OFADD__(v89, 0x7FFFFFFFFFFFFFFFLL) )
          goto LABEL_70;
        goto LABEL_118;
      }
    }
    else if ( v61 > 0 )
    {
      if ( !__OFADD__(v89, 0x7FFFFFFFFFFFFFFFLL) )
      {
        v42 = v89 + 0x7FFFFFFFFFFFFFFFLL;
        goto LABEL_70;
      }
LABEL_118:
      v42 = 0x7FFFFFFFFFFFFFFFLL;
      goto LABEL_70;
    }
    v74 = 0x8000000000000000LL;
    v42 = v89 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v89) )
      goto LABEL_70;
    goto LABEL_103;
  }
  v63 = v61 * a6;
  if ( is_mul_ok(v61, a6) )
  {
LABEL_69:
    v42 = v63 + v89;
    if ( __OFADD__(v63, v89) )
    {
      v42 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v63 <= 0 )
        v42 = 0x8000000000000000LL;
    }
    goto LABEL_70;
  }
  if ( a6 > 0 )
  {
    if ( v61 > 0 )
      goto LABEL_102;
LABEL_106:
    v74 = 0x8000000000000000LL;
    v42 = v89 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v89) )
      goto LABEL_70;
    goto LABEL_103;
  }
  if ( a6 >= 0 || v61 >= 0 )
    goto LABEL_106;
LABEL_102:
  v74 = 0x7FFFFFFFFFFFFFFFLL;
  v42 = v89 + 0x7FFFFFFFFFFFFFFFLL;
  if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v89) )
LABEL_103:
    v42 = v74;
LABEL_70:
  v43 = 1;
  v44 = 0;
LABEL_36:
  v45 = sub_3064F80(v22, (__int64)a3, (__int64 *)&v102, v44, v43);
  v46 = __OFADD__(v45, v42);
  v47 = v45 + v42;
  if ( v46 )
  {
    v47 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v45 <= 0 )
      v47 = 0x8000000000000000LL;
  }
  if ( a10 )
  {
    v64 = (__int64 *)sub_BCB2B0(*a3);
    v65 = &v100;
    if ( a11 )
      v65 = &v102;
    v66 = v64;
    v95 = (__int64 *)v65;
    v93 = sub_BCDA70(v64, v23);
    v67 = sub_BCDA70(v66, v23 * a4);
    sub_C4DEC0((__int64)&v104, (__int64)v95, v23, 0);
    v68 = sub_3064F80(v22, v93, (__int64 *)&v104, 0, 1);
    v69 = sub_3064F80(v22, v67, v95, 1, 0);
    v46 = __OFADD__(v69, v68);
    v70 = v69 + v68;
    if ( v46 )
    {
      v70 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v69 <= 0 )
        v70 = 0x8000000000000000LL;
    }
    if ( (unsigned int)v105 > 0x40 && v104 )
      j_j___libc_free_0_0((unsigned __int64)v104);
    v46 = __OFADD__(v70, v47);
    v47 += v70;
    if ( v46 )
    {
      v47 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v70 <= 0 )
        v47 = 0x8000000000000000LL;
    }
    if ( a11 )
    {
      v71 = sub_BCDA70(v66, v96);
      v72 = sub_3075ED0(v22, 28, v71, a9, 0, 0, 0, 0, 0);
      v46 = __OFADD__(v72, v47);
      v47 += v72;
      if ( v46 )
      {
        v47 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v72 <= 0 )
          v47 = 0x8000000000000000LL;
      }
    }
  }
  v11 = v47;
  if ( v103 > 0x40 && v102 )
    j_j___libc_free_0_0(v102);
  if ( v101 > 0x40 && v100 )
    j_j___libc_free_0_0(v100);
  if ( v99 > 0x40 && v98 )
    j_j___libc_free_0_0(v98);
  return v11;
}
