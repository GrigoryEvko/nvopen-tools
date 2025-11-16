// Function: sub_2B6DAD0
// Address: 0x2b6dad0
//
__int64 __fastcall sub_2B6DAD0(__int64 a1)
{
  __int64 ***v1; // rax
  __int64 **v2; // rdx
  __int64 *v3; // rax
  _QWORD *v4; // r15
  __int64 **v5; // r12
  char *v6; // rax
  char *v7; // rdx
  char v8; // al
  char v9; // cl
  char v10; // si
  unsigned int v11; // esi
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  size_t v18; // rdx
  __int64 v19; // r15
  char v20; // r12
  __int64 v21; // r13
  __int64 v22; // r10
  _QWORD *v23; // r11
  __int64 **v24; // r9
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 *v28; // rbx
  __int64 *v29; // r14
  __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int8 *v32; // rax
  int v33; // edx
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r12
  int v37; // edx
  int v38; // r15d
  __int64 v39; // rax
  int v40; // edx
  bool v41; // of
  unsigned __int64 v42; // r12
  __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // edx
  signed __int64 v46; // r12
  unsigned __int64 v47; // r14
  int v49; // eax
  __int64 v50; // rax
  unsigned int v51; // r14d
  __int64 *v52; // rbx
  __int64 *v53; // r13
  __int64 v54; // rdi
  __int64 v55; // rax
  int v56; // edx
  int v57; // edx
  unsigned __int8 *v58; // rax
  __int64 v59; // r15
  __int64 v60; // r13
  __int64 v61; // r12
  int v62; // edx
  __int64 v63; // rax
  int v64; // edx
  char v65; // al
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // r12
  unsigned __int64 *v69; // rax
  __int64 v70; // rdx
  unsigned int v71; // r14d
  __int64 v72; // rax
  unsigned __int64 *v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // ecx
  __int64 v76; // rsi
  unsigned int v77; // r12d
  __int64 v78; // rax
  bool v79; // cf
  __int64 v80; // [rsp+20h] [rbp-100h]
  __int64 **v81; // [rsp+28h] [rbp-F8h]
  _QWORD *v82; // [rsp+30h] [rbp-F0h]
  __int64 v83; // [rsp+38h] [rbp-E8h]
  __int64 v84; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v85; // [rsp+48h] [rbp-D8h]
  __int64 v86; // [rsp+50h] [rbp-D0h]
  bool v88; // [rsp+67h] [rbp-B9h]
  _QWORD *v89; // [rsp+68h] [rbp-B8h]
  __int64 v90; // [rsp+70h] [rbp-B0h]
  __int64 *v91; // [rsp+78h] [rbp-A8h]
  __int64 ***v92; // [rsp+80h] [rbp-A0h]
  _QWORD *v93; // [rsp+88h] [rbp-98h]
  unsigned __int64 v94; // [rsp+88h] [rbp-98h]
  unsigned __int64 v95; // [rsp+90h] [rbp-90h]
  unsigned __int64 v96; // [rsp+90h] [rbp-90h]
  __int64 *v97; // [rsp+98h] [rbp-88h]
  unsigned __int64 *v98; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-78h]
  __int64 *v100; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v101; // [rsp+B8h] [rbp-68h]
  _QWORD v102[12]; // [rsp+C0h] [rbp-60h] BYREF

  v1 = *(__int64 ****)(a1 + 8);
  v2 = *v1;
  v92 = v1;
  v3 = **v1;
  v91 = &v3[*((unsigned int *)v2 + 2)];
  if ( v3 == v91 )
  {
LABEL_39:
    v32 = *(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 416LL);
    v33 = *v32;
    if ( (unsigned int)(v33 - 42) <= 0x11 )
    {
      v36 = sub_DFD800(*(_QWORD *)a1, v33 - 29, **(_QWORD **)(a1 + 24), **(_DWORD **)(a1 + 32), 0, 0, 0, 0, 0, 0);
      v38 = v57;
      v39 = sub_DFD800(
              *(_QWORD *)a1,
              (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 424LL) - 29,
              **(_QWORD **)(a1 + 24),
              **(_DWORD **)(a1 + 32),
              0,
              0,
              0,
              0,
              0,
              0);
    }
    else
    {
      v34 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL);
      if ( (unsigned __int8)(***(_BYTE ***)(a1 + 40) - 82) > 1u )
      {
        if ( (v32[7] & 0x40) != 0 )
          v58 = (unsigned __int8 *)*((_QWORD *)v32 - 1);
        else
          v58 = &v32[-32 * (*((_DWORD *)v32 + 1) & 0x7FFFFFF)];
        v59 = *(_QWORD *)(*(_QWORD *)v58 + 8LL);
        v60 = sub_2B08680(v59, v34);
        if ( *(_BYTE *)(v59 + 8) != 12 || *(_BYTE *)(**(_QWORD **)(a1 + 64) + 8LL) != 12 )
          goto LABEL_73;
        v66 = *(_QWORD *)(a1 + 48);
        v67 = sub_2B68AE0(v66, **(_QWORD **)(a1 + 16), 0);
        sub_2B3B8A0(&v100, (__int64 *)(v66 + 3520), v67);
        v68 = v102[0];
        v69 = (unsigned __int64 *)sub_9208B0(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 3344LL), **(_QWORD **)(a1 + 64));
        v99 = v70;
        v98 = v69;
        v71 = sub_CA1930(&v98);
        v72 = sub_986520(*(_QWORD *)(**(_QWORD **)(a1 + 16) + 416LL));
        v73 = (unsigned __int64 *)sub_9208B0(
                                    *(_QWORD *)(*(_QWORD *)(a1 + 48) + 3344LL),
                                    *(_QWORD *)(*(_QWORD *)v72 + 8LL));
        v99 = v74;
        v98 = v73;
        v75 = sub_CA1930(&v98);
        if ( v68 == *(_QWORD *)(*(_QWORD *)(a1 + 48) + 3528LL) + 24LL * *(unsigned int *)(*(_QWORD *)(a1 + 48) + 3544LL) )
        {
          v77 = v75;
        }
        else
        {
          v76 = *(_QWORD *)(v68 + 8);
          v77 = v76;
          v78 = sub_BCCE00(*(_QWORD **)v59, v76);
          v60 = sub_2B08680(v78, *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL));
        }
        v79 = v71 < v77;
        if ( v71 > v77 )
        {
LABEL_73:
          v61 = sub_DFD060(
                  *(__int64 **)a1,
                  (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 416LL) - 29,
                  **(_QWORD **)(a1 + 24),
                  v60);
          v38 = v62;
          v63 = sub_DFD060(
                  *(__int64 **)a1,
                  (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 424LL) - 29,
                  **(_QWORD **)(a1 + 24),
                  v60);
          if ( v64 == 1 )
            v38 = 1;
          v41 = __OFADD__(v63, v61);
          v42 = v63 + v61;
          if ( v41 )
          {
            v42 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v63 <= 0 )
              v42 = 0x8000000000000000LL;
          }
          goto LABEL_45;
        }
        v46 = 0;
        if ( v79 )
          return sub_DFD060(*(__int64 **)a1, 38, **(_QWORD **)(a1 + 24), v60);
        return v46;
      }
      v35 = sub_BCB2A0(*(_QWORD **)(*(_QWORD *)(a1 + 48) + 3440LL));
      sub_2B08680(v35, v34);
      v36 = sub_DFD2D0(
              *(__int64 **)a1,
              (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 416LL) - 29,
              **(_QWORD **)(a1 + 24));
      v38 = v37;
      v39 = sub_DFD2D0(
              *(__int64 **)a1,
              (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 416LL) - 29,
              **(_QWORD **)(a1 + 24));
    }
    if ( v40 == 1 )
      v38 = 1;
    v41 = __OFADD__(v39, v36);
    v42 = v39 + v36;
    if ( v41 )
    {
      v42 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v39 <= 0 )
        v42 = 0x8000000000000000LL;
    }
    goto LABEL_45;
  }
  v97 = v3;
LABEL_5:
  v4 = (_QWORD *)*v97;
  v5 = v92[1];
  if ( (__int64 **)*v97 == v5 )
    goto LABEL_39;
  v6 = (char *)v4[52];
  if ( !v6 )
    goto LABEL_4;
  v7 = (char *)v4[53];
  v88 = v7 != 0 && v6 != v7;
  if ( !v88 )
    goto LABEL_4;
  v8 = *v6;
  v9 = *(_BYTE *)v5[52];
  v10 = *(_BYTE *)v5[53];
  if ( (v8 != v9 || *v7 != v10) && (v8 != v10 || v9 != *v7) )
    goto LABEL_4;
  v11 = *((_DWORD *)v4 + 62);
  if ( *((_DWORD *)v5 + 62) != v11 )
    goto LABEL_4;
  sub_B48880((__int64 *)&v100, v11, 0);
  v12 = *((unsigned int *)v4 + 62);
  v13 = (unsigned __int64)v100;
  if ( (_DWORD)v12 )
  {
    v83 = *((unsigned int *)v4 + 62);
    v82 = v4;
    v80 = 80 * v12;
    v81 = v5;
    v14 = 0;
LABEL_22:
    v85 = v13 >> 58;
    v84 = ~(-1LL << (v13 >> 58));
    v95 = v84 & (v13 >> 1);
    v20 = v13 & 1;
    if ( (v13 & 1) != 0 )
    {
      v86 = (int)sub_39FAC40(~(-1LL << (v13 >> 58)) & (v13 >> 1));
    }
    else
    {
      v50 = *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8);
      if ( *(_QWORD *)v13 == v50 )
      {
        v86 = 0;
      }
      else
      {
        v94 = v13;
        v51 = 0;
        v52 = *(__int64 **)v13;
        v53 = (__int64 *)v50;
        do
        {
          v54 = *v52++;
          v51 += sub_39FAC40(v54);
        }
        while ( v53 != v52 );
        v13 = v94;
        v86 = v51;
      }
    }
    v22 = v14;
    v23 = (_QWORD *)v13;
    v24 = v81;
    v25 = 0;
    while ( 1 )
    {
      if ( v20 )
      {
        if ( ((v95 >> v25) & 1) == 0 )
          goto LABEL_14;
LABEL_27:
        if ( v83 == ++v25 )
          goto LABEL_28;
      }
      else
      {
        if ( ((*(_QWORD *)(*v23 + 8LL * ((unsigned int)v25 >> 6)) >> v25) & 1) != 0 )
          goto LABEL_27;
LABEL_14:
        v15 = v82[30] + 80 * v25;
        v16 = v22 + v24[30];
        v17 = *(unsigned int *)(v15 + 8);
        if ( *(_DWORD *)(v16 + 8) != (_DWORD)v17 )
          goto LABEL_27;
        v18 = 8 * v17;
        v93 = v24;
        if ( !v18
          || (v89 = v23,
              v90 = v22,
              v49 = memcmp(*(const void **)v15, *(const void **)v16, v18),
              v22 = v90,
              v23 = v89,
              v24 = v93,
              !v49) )
        {
          v19 = v22;
          if ( v20 )
          {
            v13 = 2 * ((v85 << 57) | v84 & (v95 | (1LL << v25))) + 1;
            v100 = (__int64 *)v13;
            v20 = (2 * (v84 & (v95 | (1 << v25))) + 1) & 1;
          }
          else
          {
            *(_QWORD *)(*v23 + 8LL * ((unsigned int)v25 >> 6)) |= 1LL << v25;
            v13 = (unsigned __int64)v100;
            v20 = (unsigned __int8)v100 & 1;
          }
          if ( v20 )
          {
LABEL_19:
            v21 = (int)sub_39FAC40((v13 >> 1) & ~(-1LL << (v13 >> 58)));
            goto LABEL_20;
          }
LABEL_29:
          v26 = *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8);
          if ( *(_QWORD *)v13 != v26 )
          {
            v96 = v13;
            LODWORD(v27) = 0;
            v28 = *(__int64 **)v13;
            v29 = (__int64 *)v26;
            do
            {
              v30 = *v28++;
              v27 = (unsigned int)sub_39FAC40(v30) + (unsigned int)v27;
            }
            while ( v29 != v28 );
            v13 = v96;
            if ( v86 == v27 )
              goto LABEL_33;
LABEL_21:
            v14 = v19 + 80;
            if ( v80 != v14 )
              goto LABEL_22;
            v65 = v20;
            goto LABEL_84;
          }
          v21 = 0;
LABEL_20:
          if ( v86 != v21 )
            goto LABEL_21;
LABEL_33:
          if ( !v20 && v13 )
          {
            v31 = *(_QWORD *)v13;
            v88 = 0;
            if ( *(_QWORD *)v13 != v13 + 16 )
              goto LABEL_87;
            j_j___libc_free_0(v13);
          }
LABEL_4:
          if ( v91 == ++v97 )
            goto LABEL_39;
          goto LABEL_5;
        }
        if ( v83 == ++v25 )
        {
LABEL_28:
          v19 = v22;
          v13 = (unsigned __int64)v23;
          if ( v20 )
            goto LABEL_19;
          goto LABEL_29;
        }
      }
    }
  }
  v65 = (unsigned __int8)v100 & 1;
LABEL_84:
  if ( v65 || !v13 )
  {
LABEL_88:
    v38 = 0;
    v42 = 0;
    goto LABEL_45;
  }
  v31 = *(_QWORD *)v13;
  if ( *(_QWORD *)v13 != v13 + 16 )
  {
LABEL_87:
    _libc_free(v31);
    j_j___libc_free_0(v13);
    if ( !v88 )
      goto LABEL_4;
    goto LABEL_88;
  }
  v38 = 0;
  v42 = 0;
  j_j___libc_free_0(v13);
LABEL_45:
  v100 = v102;
  v101 = 0xC00000000LL;
  v43 = **(_QWORD **)(a1 + 16);
  v98 = *(unsigned __int64 **)(a1 + 16);
  v99 = *(_QWORD *)(a1 + 48);
  sub_2B315C0(v43, (__int64 (__fastcall *)(__int64, _BYTE *))sub_2B64FB0, (__int64)&v98, (unsigned __int64)&v100, 0, 0);
  v44 = sub_2B097B0(
          *(__int64 **)a1,
          6,
          **(_QWORD **)(a1 + 72),
          (int *)v100,
          (unsigned int)v101,
          **(unsigned int **)(a1 + 32),
          0,
          0);
  if ( v45 == 1 )
    v38 = 1;
  v41 = __OFADD__(v44, v42);
  v46 = v44 + v42;
  if ( v41 )
  {
    v46 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v44 <= 0 )
      v46 = 0x8000000000000000LL;
  }
  sub_2B24BA0(
    (__int64 *)&v98,
    ***(_QWORD ***)(a1 + 16),
    *(_DWORD *)(**(_QWORD **)(a1 + 16) + 8LL),
    **(unsigned __int8 **)(**(_QWORD **)(a1 + 16) + 424LL) - 29);
  if ( (unsigned __int8)sub_DFA540(*(_QWORD *)a1) )
  {
    v55 = sub_DFBBE0(*(_QWORD *)a1);
    if ( v56 == v38 )
    {
      if ( v46 <= v55 )
        goto LABEL_49;
    }
    else if ( v56 >= v38 )
    {
      goto LABEL_49;
    }
    v46 = v55;
  }
LABEL_49:
  v47 = (unsigned __int64)v98;
  if ( ((unsigned __int8)v98 & 1) == 0 && v98 )
  {
    if ( (unsigned __int64 *)*v98 != v98 + 2 )
      _libc_free(*v98);
    j_j___libc_free_0(v47);
  }
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  return v46;
}
