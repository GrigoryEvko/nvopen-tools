// Function: sub_2859900
// Address: 0x2859900
//
void __fastcall sub_2859900(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 i; // r12
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  unsigned __int64 *v20; // r12
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  _BYTE *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rbx
  unsigned int *v36; // r12
  __int64 v37; // rax
  __int64 v38; // r13
  _BYTE *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // r12
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // rbx
  __int64 v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rcx
  const void *v50; // rsi
  __int64 v51; // rax
  __int64 *v52; // r12
  __int64 v53; // rbx
  __int64 v54; // rsi
  __int64 v55; // rax
  _QWORD *v56; // rdi
  _QWORD *v57; // rdx
  __int64 v58; // rcx
  _QWORD *v59; // rsi
  _QWORD *v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 j; // rdx
  _BYTE *v65; // rbx
  unsigned __int64 v66; // r12
  unsigned int v67; // eax
  __int64 v68; // rbx
  unsigned __int64 v69; // r12
  _BYTE *v70; // r13
  _QWORD *v71; // rdi
  _QWORD *v72; // rdx
  _QWORD *v73; // rax
  __int64 v74; // rsi
  _BYTE **v75; // rdi
  unsigned __int64 *v76; // rax
  unsigned __int64 *v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 *v81; // r13
  __int64 v82; // rbx
  unsigned __int64 *v83; // r15
  unsigned __int64 v84; // r14
  unsigned __int64 *v85; // rax
  char v86; // dl
  __int64 *v87; // rax
  unsigned __int64 *v88; // r12
  unsigned __int64 *v89; // rbx
  unsigned __int64 v90; // [rsp+10h] [rbp-510h]
  __int64 v91; // [rsp+10h] [rbp-510h]
  __int64 v92; // [rsp+18h] [rbp-508h]
  int v93; // [rsp+20h] [rbp-500h]
  __int64 v94; // [rsp+20h] [rbp-500h]
  unsigned int v95; // [rsp+28h] [rbp-4F8h]
  __int64 v96; // [rsp+28h] [rbp-4F8h]
  _BYTE *v97; // [rsp+30h] [rbp-4F0h]
  int v98; // [rsp+30h] [rbp-4F0h]
  __int64 v99; // [rsp+30h] [rbp-4F0h]
  _BOOL4 v100; // [rsp+38h] [rbp-4E8h]
  unsigned int v101; // [rsp+38h] [rbp-4E8h]
  __int64 v102; // [rsp+40h] [rbp-4E0h]
  __int64 v103; // [rsp+48h] [rbp-4D8h]
  unsigned int v104; // [rsp+48h] [rbp-4D8h]
  __int64 v105; // [rsp+50h] [rbp-4D0h] BYREF
  unsigned __int64 *v106; // [rsp+58h] [rbp-4C8h]
  __int64 v107; // [rsp+60h] [rbp-4C0h]
  int v108; // [rsp+68h] [rbp-4B8h]
  char v109; // [rsp+6Ch] [rbp-4B4h]
  char v110; // [rsp+70h] [rbp-4B0h] BYREF
  _BYTE *v111; // [rsp+90h] [rbp-490h] BYREF
  __int64 v112; // [rsp+98h] [rbp-488h]
  _BYTE v113[64]; // [rsp+A0h] [rbp-480h] BYREF
  _BYTE *v114; // [rsp+E0h] [rbp-440h] BYREF
  __int64 v115; // [rsp+E8h] [rbp-438h]
  _BYTE v116[1072]; // [rsp+F0h] [rbp-430h] BYREF

  v1 = a1;
  v2 = *(_QWORD *)(a1 + 56);
  v114 = v116;
  v3 = *(_QWORD *)(v1 + 16);
  v115 = 0x800000000LL;
  v112 = 0x800000000LL;
  v4 = *(__int64 **)(v2 + 32);
  v111 = v113;
  v5 = *v4;
  v6 = sub_D47930(v2);
  if ( v6 )
  {
    v9 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
    v10 = *(_DWORD *)(v6 + 44) + 1;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  if ( *(_DWORD *)(v3 + 32) <= v10 )
    BUG();
  v11 = *(__int64 **)(*(_QWORD *)(v3 + 24) + 8 * v9);
  v12 = (unsigned int)v112;
  for ( i = *v11; v5 != *v11; i = *v11 )
  {
    if ( v12 + 1 > (unsigned __int64)HIDWORD(v112) )
    {
      sub_C8D5F0((__int64)&v111, v113, v12 + 1, 8u, v7, v8);
      v12 = (unsigned int)v112;
    }
    *(_QWORD *)&v111[8 * v12] = i;
    v12 = (unsigned int)(v112 + 1);
    LODWORD(v112) = v112 + 1;
    v11 = (__int64 *)v11[1];
  }
  if ( v12 + 1 > (unsigned __int64)HIDWORD(v112) )
  {
    sub_C8D5F0((__int64)&v111, v113, v12 + 1, 8u, v7, v8);
    v12 = (unsigned int)v112;
  }
  *(_QWORD *)&v111[8 * v12] = v5;
  LODWORD(v112) = v112 + 1;
  v90 = (unsigned __int64)v111;
  v97 = &v111[8 * (unsigned int)v112];
  if ( v97 != v111 )
  {
    while ( 1 )
    {
      v14 = *((_QWORD *)v97 - 1);
      v15 = *(_QWORD *)(v14 + 56);
      v103 = v14 + 48;
      if ( v14 + 48 != v15 )
        break;
LABEL_22:
      v97 -= 8;
      if ( (_BYTE *)v90 == v97 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      if ( !v15 )
        BUG();
      v16 = v15 - 24;
      if ( *(_BYTE *)(v15 - 24) != 84 )
      {
        v17 = *(_QWORD *)v1;
        if ( *(_BYTE *)(*(_QWORD *)v1 + 68LL) )
        {
          v18 = *(_QWORD **)(v17 + 48);
          v19 = &v18[*(unsigned int *)(v17 + 60)];
          if ( v18 == v19 )
            goto LABEL_21;
          while ( v16 != *v18 )
          {
            if ( v19 == ++v18 )
              goto LABEL_21;
          }
        }
        else if ( !sub_C8CA60(v17 + 40, v15 - 24) )
        {
          goto LABEL_21;
        }
        v20 = (unsigned __int64 *)(v15 - 24);
        if ( !sub_D97040(*(_QWORD *)(v1 + 8), *(_QWORD *)(v15 - 16))
          || *((_WORD *)sub_DD8400(*(_QWORD *)(v1 + 8), v15 - 24) + 12) == 15 )
        {
          break;
        }
      }
LABEL_21:
      v15 = *(_QWORD *)(v15 + 8);
      if ( v103 == v15 )
        goto LABEL_22;
    }
    v67 = *(_DWORD *)(v1 + 36464);
    if ( v67 )
    {
      v94 = v15;
      v68 = 0;
      v69 = (unsigned __int64)v67 << 7;
      do
      {
        v70 = &v114[v68];
        if ( v114[v68 + 92] )
        {
          v71 = (_QWORD *)*((_QWORD *)v70 + 9);
          v72 = &v71[*((unsigned int *)v70 + 21)];
          v73 = v71;
          if ( v71 != v72 )
          {
            while ( v16 != *v73 )
            {
              if ( v72 == ++v73 )
                goto LABEL_106;
            }
            v74 = (unsigned int)(*((_DWORD *)v70 + 21) - 1);
            *((_DWORD *)v70 + 21) = v74;
            *v73 = v71[v74];
            ++*((_QWORD *)v70 + 8);
          }
        }
        else
        {
          v87 = sub_C8CA60((__int64)(v70 + 64), v16);
          if ( v87 )
          {
            *v87 = -2;
            ++*((_DWORD *)v70 + 22);
            ++*((_QWORD *)v70 + 8);
          }
        }
LABEL_106:
        v68 += 128;
      }
      while ( v68 != v69 );
      v20 = (unsigned __int64 *)v16;
      v15 = v94;
    }
    v105 = 0;
    v107 = 4;
    v106 = (unsigned __int64 *)&v110;
    v108 = 0;
    v109 = 1;
    if ( (*(_BYTE *)(v15 - 17) & 0x40) != 0 )
    {
      v75 = *(_BYTE ***)(v15 - 32);
      v20 = (unsigned __int64 *)&v75[4 * (*(_DWORD *)(v15 - 20) & 0x7FFFFFF)];
    }
    else
    {
      v75 = (_BYTE **)(v16 - 32LL * (*(_DWORD *)(v15 - 20) & 0x7FFFFFF));
    }
    v76 = (unsigned __int64 *)sub_284F450(v75, (_BYTE **)v20, *(_QWORD *)(v1 + 56), *(_QWORD *)(v1 + 8));
    if ( v76 == v20 )
      goto LABEL_119;
    v96 = v15;
    v81 = (__int64 *)v16;
    v82 = v1;
    v83 = v76;
    while ( 1 )
    {
      v84 = *v83;
      if ( v109 )
      {
        v85 = v106;
        v78 = HIDWORD(v107);
        v77 = &v106[HIDWORD(v107)];
        if ( v106 != v77 )
        {
          while ( v84 != *v85 )
          {
            if ( v77 == ++v85 )
              goto LABEL_123;
          }
          goto LABEL_117;
        }
LABEL_123:
        if ( HIDWORD(v107) < (unsigned int)v107 )
          break;
      }
      sub_C8CC70((__int64)&v105, *v83, (__int64)v77, v78, v79, v80);
      if ( v86 )
        goto LABEL_122;
LABEL_117:
      v83 = (unsigned __int64 *)sub_284F450(
                                  (_BYTE **)v83 + 4,
                                  (_BYTE **)v20,
                                  *(_QWORD *)(v82 + 56),
                                  *(_QWORD *)(v82 + 8));
      if ( v83 == v20 )
      {
        v1 = v82;
        v15 = v96;
LABEL_119:
        if ( !v109 )
          _libc_free((unsigned __int64)v106);
        goto LABEL_21;
      }
    }
    ++HIDWORD(v107);
    *v77 = v84;
    ++v105;
LABEL_122:
    sub_2858C90(v82, v81, v84, (__int64)&v114);
    goto LABEL_117;
  }
LABEL_23:
  v21 = sub_AA5930(**(_QWORD **)(*(_QWORD *)(v1 + 56) + 32LL));
  v25 = v24;
  v26 = v21;
  while ( v25 != v26 )
  {
    if ( sub_D97040(*(_QWORD *)(v1 + 8), *(_QWORD *)(v26 + 8)) )
    {
      v27 = sub_D47930(*(_QWORD *)(v1 + 56));
      v28 = *(_QWORD *)(v26 - 8);
      v29 = v27;
      if ( (*(_DWORD *)(v26 + 4) & 0x7FFFFFF) != 0 )
      {
        v30 = 0;
        while ( v29 != *(_QWORD *)(v28 + 32LL * *(unsigned int *)(v26 + 72) + 8 * v30) )
        {
          if ( (*(_DWORD *)(v26 + 4) & 0x7FFFFFF) == (_DWORD)++v30 )
            goto LABEL_134;
        }
        v31 = 32 * v30;
      }
      else
      {
LABEL_134:
        v31 = 0x1FFFFFFFE0LL;
      }
      v32 = *(_BYTE **)(v28 + v31);
      if ( *v32 > 0x1Cu )
        sub_2858C90(v1, (__int64 *)v26, (unsigned __int64)v32, (__int64)&v114);
    }
    v33 = *(_QWORD *)(v26 + 32);
    if ( !v33 )
      BUG();
    v26 = 0;
    if ( *(_BYTE *)(v33 - 24) == 84 )
      v26 = v33 - 24;
  }
  if ( !*(_DWORD *)(v1 + 36464) )
    goto LABEL_83;
  v102 = *(unsigned int *)(v1 + 36464);
  v34 = 0;
  v104 = 0;
  do
  {
    v35 = 48 * v34;
    v36 = (unsigned int *)(48 * v34 + *(_QWORD *)(v1 + 36456));
    v37 = v36[2];
    if ( (unsigned int)v37 <= 1 || *(_DWORD *)&v114[128 * v34 + 20] != *(_DWORD *)&v114[128 * v34 + 24] )
      goto LABEL_73;
    v38 = *(_QWORD *)(v1 + 48);
    v100 = 1;
    v39 = *(_BYTE **)(*(_QWORD *)v36 + 24 * v37 - 24);
    if ( *v39 == 84 )
      v100 = *(_QWORD *)(*(_QWORD *)v36 + 16LL) != (_QWORD)sub_DD8400(*(_QWORD *)(v1 + 8), (__int64)v39);
    if ( (unsigned __int8)sub_DFA2A0(v38) )
      goto LABEL_53;
    v40 = *(_QWORD *)v36 + 24LL * v36[2];
    if ( v40 == *(_QWORD *)v36 + 24LL )
      goto LABEL_73;
    v92 = v34;
    v41 = *(_QWORD *)v36 + 24LL;
    v91 = v35;
    v42 = 0;
    v43 = v40;
    v98 = 0;
    v93 = 0;
    v95 = 0;
    do
    {
      while ( 1 )
      {
        if ( (unsigned __int8)sub_DFA2A0(v38) )
        {
          v34 = v92;
          v35 = v91;
LABEL_53:
          if ( v104 != (_DWORD)v34 )
          {
            v45 = *(_QWORD *)(v1 + 36456) + v35;
            v46 = 48LL * v104 + *(_QWORD *)(v1 + 36456);
            if ( v46 != v45 )
            {
              v48 = *(unsigned int *)(v46 + 8);
              v101 = *(_DWORD *)(v45 + 8);
              v47 = v101;
              if ( v101 <= v48 )
              {
                if ( v101 )
                  memmove(*(void **)v46, *(const void **)v45, 24LL * v101);
              }
              else
              {
                if ( v101 > (unsigned __int64)*(unsigned int *)(v46 + 12) )
                {
                  *(_DWORD *)(v46 + 8) = 0;
                  sub_C8D5F0(v46, (const void *)(v46 + 16), v101, 0x18u, v22, v23);
                  v47 = *(unsigned int *)(v45 + 8);
                  v49 = 0;
                }
                else
                {
                  v49 = 24 * v48;
                  if ( *(_DWORD *)(v46 + 8) )
                  {
                    v99 = 24 * v48;
                    memmove(*(void **)v46, *(const void **)v45, 24 * v48);
                    v47 = *(unsigned int *)(v45 + 8);
                    v49 = v99;
                  }
                }
                v50 = (const void *)(*(_QWORD *)v45 + v49);
                if ( v50 != (const void *)(24 * v47 + *(_QWORD *)v45) )
                  memcpy((void *)(v49 + *(_QWORD *)v46), v50, 24 * v47 - v49);
              }
              *(_DWORD *)(v46 + 8) = v101;
            }
            *(_QWORD *)(v46 + 40) = *(_QWORD *)(v45 + 40);
          }
          v51 = 48LL * v104 + *(_QWORD *)(v1 + 36456);
          v52 = (__int64 *)(*(_QWORD *)v51 + 24LL);
          v53 = *(_QWORD *)v51 + 24LL * *(unsigned int *)(v51 + 8);
          if ( (__int64 *)v53 == v52 )
          {
LABEL_72:
            ++v104;
            goto LABEL_73;
          }
          while ( 1 )
          {
            v54 = *v52;
            v55 = 4LL * (*(_DWORD *)(*v52 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(*v52 + 7) & 0x40) != 0 )
            {
              v56 = *(_QWORD **)(v54 - 8);
              v54 = (__int64)&v56[v55];
            }
            else
            {
              v56 = (_QWORD *)(v54 - v55 * 8);
            }
            v59 = sub_284FF00(v56, v54, v52 + 1);
            if ( !*(_BYTE *)(v1 + 36884) )
              goto LABEL_128;
            v60 = *(_QWORD **)(v1 + 36864);
            v58 = *(unsigned int *)(v1 + 36876);
            v57 = &v60[v58];
            if ( v60 == v57 )
            {
LABEL_131:
              if ( (unsigned int)v58 >= *(_DWORD *)(v1 + 36872) )
              {
LABEL_128:
                v52 += 3;
                sub_C8CC70(v1 + 36856, (__int64)v59, (__int64)v57, v58, v22, v23);
                if ( (__int64 *)v53 == v52 )
                  goto LABEL_72;
              }
              else
              {
                v52 += 3;
                *(_DWORD *)(v1 + 36876) = v58 + 1;
                *v57 = v59;
                ++*(_QWORD *)(v1 + 36856);
                if ( (__int64 *)v53 == v52 )
                  goto LABEL_72;
              }
            }
            else
            {
              while ( v59 != (_QWORD *)*v60 )
              {
                if ( v57 == ++v60 )
                  goto LABEL_131;
              }
              v52 += 3;
              if ( (__int64 *)v53 == v52 )
                goto LABEL_72;
            }
          }
        }
        if ( !sub_D968A0(*(_QWORD *)(v41 + 16)) )
          break;
LABEL_47:
        v41 += 24;
        if ( v43 == v41 )
          goto LABEL_52;
      }
      v44 = *(_QWORD *)(v41 + 16);
      if ( *(_WORD *)(v44 + 24) )
      {
        if ( v42 == v44 )
        {
          ++v93;
        }
        else
        {
          ++v98;
          v42 = *(_QWORD *)(v41 + 16);
        }
        goto LABEL_47;
      }
      v41 += 24;
      ++v95;
    }
    while ( v43 != v41 );
LABEL_52:
    v34 = v92;
    v35 = v91;
    if ( v98 - v93 + (v95 < 2) + v100 - 1 < 0 )
      goto LABEL_53;
LABEL_73:
    ++v34;
  }
  while ( v34 != v102 );
  v61 = *(unsigned int *)(v1 + 36464);
  if ( v61 != v104 )
  {
    v62 = 48LL * v104;
    if ( v61 > v104 )
    {
      v88 = (unsigned __int64 *)(*(_QWORD *)(v1 + 36456) + v62);
      v89 = (unsigned __int64 *)(*(_QWORD *)(v1 + 36456) + 48 * v61);
      while ( v88 != v89 )
      {
        v89 -= 6;
        if ( (unsigned __int64 *)*v89 != v89 + 2 )
          _libc_free(*v89);
      }
    }
    else
    {
      if ( v104 > *(_DWORD *)(v1 + 36468) )
      {
        sub_2850DE0(v1 + 36456, v104, 48LL * v104, v104, v22, v23);
        v61 = *(unsigned int *)(v1 + 36464);
      }
      v63 = *(_QWORD *)(v1 + 36456) + 48 * v61;
      for ( j = v62 + *(_QWORD *)(v1 + 36456); j != v63; v63 += 48 )
      {
        if ( v63 )
        {
          *(_DWORD *)(v63 + 8) = 0;
          *(_QWORD *)v63 = v63 + 16;
          *(_DWORD *)(v63 + 12) = 1;
          *(_OWORD *)(v63 + 16) = 0;
          *(_OWORD *)(v63 + 32) = 0;
        }
      }
    }
    *(_DWORD *)(v1 + 36464) = v104;
  }
LABEL_83:
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
  v65 = v114;
  v66 = (unsigned __int64)&v114[128 * (unsigned __int64)(unsigned int)v115];
  if ( v114 != (_BYTE *)v66 )
  {
    while ( 1 )
    {
      v66 -= 128LL;
      if ( *(_BYTE *)(v66 + 92) )
      {
        if ( *(_BYTE *)(v66 + 28) )
          goto LABEL_88;
LABEL_91:
        _libc_free(*(_QWORD *)(v66 + 8));
        if ( v65 == (_BYTE *)v66 )
        {
LABEL_92:
          v66 = (unsigned __int64)v114;
          break;
        }
      }
      else
      {
        _libc_free(*(_QWORD *)(v66 + 72));
        if ( !*(_BYTE *)(v66 + 28) )
          goto LABEL_91;
LABEL_88:
        if ( v65 == (_BYTE *)v66 )
          goto LABEL_92;
      }
    }
  }
  if ( (_BYTE *)v66 != v116 )
    _libc_free(v66);
}
