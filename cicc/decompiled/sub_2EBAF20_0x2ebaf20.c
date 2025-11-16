// Function: sub_2EBAF20
// Address: 0x2ebaf20
//
void __fastcall sub_2EBAF20(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r8
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned int v13; // r12d
  unsigned __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int v21; // r14d
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // rcx
  __int64 *v30; // rdi
  __int64 *v31; // r13
  __int64 *v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rax
  unsigned int v36; // r12d
  __int64 *v37; // rdi
  int v38; // esi
  unsigned int v39; // edx
  __int64 v40; // rax
  unsigned int v41; // esi
  unsigned int v42; // edx
  unsigned int v43; // edi
  bool v44; // cf
  __int64 v45; // r12
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // r12
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rdi
  int v58; // r11d
  __int64 *v59; // r10
  __int64 *v60; // r12
  __int64 *v61; // rbx
  __int64 v62; // rdi
  __int64 *v63; // rbx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 *v71; // rax
  __int64 v72; // r10
  __int64 v73; // rax
  __int64 v74; // r11
  __int64 i; // r8
  __int64 v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rsi
  unsigned __int64 v79; // rdi
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 *v83; // rdi
  __int64 *v84; // rsi
  _QWORD *v85; // r8
  __int64 v86; // [rsp+8h] [rbp-298h]
  unsigned int v88; // [rsp+64h] [rbp-23Ch]
  __int64 v89; // [rsp+68h] [rbp-238h] BYREF
  __int64 v90; // [rsp+70h] [rbp-230h] BYREF
  __int64 *v91; // [rsp+78h] [rbp-228h] BYREF
  _BYTE *v92; // [rsp+80h] [rbp-220h] BYREF
  __int64 v93; // [rsp+88h] [rbp-218h]
  _BYTE v94[64]; // [rsp+90h] [rbp-210h] BYREF
  __int64 *v95; // [rsp+D0h] [rbp-1D0h] BYREF
  int v96; // [rsp+D8h] [rbp-1C8h]
  _BYTE v97[64]; // [rsp+E0h] [rbp-1C0h] BYREF
  _BYTE *v98; // [rsp+120h] [rbp-180h] BYREF
  __int64 v99; // [rsp+128h] [rbp-178h]
  _BYTE v100[72]; // [rsp+130h] [rbp-170h] BYREF
  __int64 v101; // [rsp+178h] [rbp-128h] BYREF
  __int64 v102; // [rsp+180h] [rbp-120h]
  __int64 *v103; // [rsp+188h] [rbp-118h] BYREF
  unsigned int v104; // [rsp+190h] [rbp-110h]
  __int64 *v105; // [rsp+1C8h] [rbp-D8h] BYREF
  __int64 v106; // [rsp+1D0h] [rbp-D0h]
  _BYTE v107[64]; // [rsp+1D8h] [rbp-C8h] BYREF
  _BYTE *v108; // [rsp+218h] [rbp-88h] BYREF
  __int64 v109; // [rsp+220h] [rbp-80h]
  _BYTE v110[120]; // [rsp+228h] [rbp-78h] BYREF

  v6 = a3;
  v9 = *(_QWORD **)(a4 + 8);
  v89 = a4;
  if ( !*v9 )
  {
    v83 = *(__int64 **)a1;
    v98 = *(_BYTE **)a4;
    v84 = &v83[*(unsigned int *)(a1 + 8)];
    if ( v84 != sub_2EB3010(v83, (__int64)v84, (__int64 *)&v98) )
    {
      sub_2EBA1B0(a1, a2);
      return;
    }
  }
  if ( *v6 && *(_QWORD *)a4 && (v10 = sub_2EB3BB0(a1, *v6, *(_QWORD *)a4)) != 0 )
  {
    v11 = (unsigned int)(*(_DWORD *)(v10 + 24) + 1);
    v12 = *(_DWORD *)(v10 + 24) + 1;
  }
  else
  {
    v11 = 0;
    v12 = 0;
  }
  if ( *(_DWORD *)(a1 + 56) <= v12 )
LABEL_113:
    BUG();
  v86 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v11);
  v13 = *(_DWORD *)(v86 + 16) + 1;
  if ( v13 >= *(_DWORD *)(a4 + 16) )
    return;
  v101 = 0;
  v98 = v100;
  v99 = 0x800000000LL;
  v14 = (unsigned __int64 *)&v103;
  v102 = 1;
  do
    *v14++ = -4096;
  while ( v14 != (unsigned __int64 *)&v105 );
  v105 = (__int64 *)v107;
  v108 = v110;
  v106 = 0x800000000LL;
  v109 = 0x800000000LL;
  v93 = 0x800000000LL;
  v92 = v94;
  sub_2E6D8E0((__int64)&v98, a4, (__int64)&v105, a4, (__int64)v6, a6);
  v15 = (__int64)v98;
  v16 = 8LL * (unsigned int)v99;
  v17 = *(_QWORD *)&v98[v16 - 8];
  v18 = (v16 >> 3) - 1;
  v19 = ((v16 >> 3) - 2) / 2;
  if ( v18 > 0 )
  {
    while ( 1 )
    {
      v20 = v15 + 8 * v19;
      v85 = (_QWORD *)(v15 + 8 * v18);
      if ( *(_DWORD *)(*(_QWORD *)v20 + 16LL) >= *(_DWORD *)(v17 + 16) )
        break;
      *v85 = *(_QWORD *)v20;
      v18 = v19;
      if ( v19 <= 0 )
      {
        v85 = (_QWORD *)(v15 + 8 * v19);
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
  else
  {
    v85 = &v98[v16 - 8];
  }
  *v85 = v17;
  v21 = v13;
  sub_2E73000((__int64)&v95, (__int64)&v101, &v89);
  v25 = (unsigned int)v99;
  if ( !(_DWORD)v99 )
    goto LABEL_67;
  do
  {
    v26 = (__int64)v98;
    v27 = 8 * v25;
    v28 = *(_QWORD *)v98;
    if ( v25 == 1 )
      goto LABEL_19;
    v71 = (__int64 *)&v98[v27 - 8];
    v72 = *v71;
    *v71 = v28;
    v73 = v27 - 8;
    v74 = (__int64)(v27 - 8) >> 3;
    v27 = v74 - 1;
    v24 = (v74 - 1) / 2;
    if ( v73 > 16 )
    {
      for ( i = 0; ; i = v76 )
      {
        v76 = 2 * (i + 1);
        v77 = v26 + 16 * (i + 1);
        v78 = *(_QWORD *)v77;
        if ( *(_DWORD *)(*(_QWORD *)v77 + 16LL) < *(_DWORD *)(*(_QWORD *)(v77 - 8) + 16LL) )
        {
          --v76;
          v77 = v26 + 8 * v76;
          v78 = *(_QWORD *)v77;
        }
        *(_QWORD *)(v26 + 8 * i) = v78;
        if ( v76 >= v24 )
          break;
      }
      if ( (v74 & 1) != 0 )
      {
LABEL_105:
        v23 = v76;
        v27 = (v76 - 1) >> 1;
LABEL_95:
        while ( 1 )
        {
          v79 = v26 + 8 * v27;
          v24 = *(unsigned int *)(v72 + 16);
          v80 = (_QWORD *)(v26 + 8 * v23);
          if ( *(_DWORD *)(*(_QWORD *)v79 + 16LL) >= (unsigned int)v24 )
            goto LABEL_96;
          *v80 = *(_QWORD *)v79;
          v23 = v27;
          if ( !v27 )
          {
            *(_QWORD *)v79 = v72;
            goto LABEL_19;
          }
          v27 = (__int64)(v27 - 1) / 2;
        }
      }
      v23 = v76;
      v27 = (v76 - 1) >> 1;
      if ( v76 != (v74 - 2) / 2 )
        goto LABEL_95;
LABEL_104:
      v81 = 2 * v76 + 2;
      v82 = *(_QWORD *)(v26 + 8 * v81 - 8);
      v76 = v81 - 1;
      *(_QWORD *)v77 = v82;
      goto LABEL_105;
    }
    v80 = (_QWORD *)v26;
    if ( (v74 & 1) == 0 && v27 <= 2 )
    {
      v77 = v26;
      v76 = 0;
      goto LABEL_104;
    }
LABEL_96:
    *v80 = v72;
LABEL_19:
    LODWORD(v99) = v99 - 1;
    sub_2E6D8E0((__int64)&v105, v28, v26, v27, v23, v24);
    v88 = *(_DWORD *)(v28 + 16);
    while ( 2 )
    {
      sub_2EB52F0(&v95, *(_QWORD *)v28, a2, v29, v23, v24);
      v30 = v95;
      v31 = &v95[v96];
      if ( v31 == v95 )
        goto LABEL_34;
      v32 = v95;
      do
      {
        v40 = *v32;
        if ( *v32 )
        {
          v33 = (unsigned int)(*(_DWORD *)(v40 + 24) + 1);
          v34 = *(_DWORD *)(v40 + 24) + 1;
        }
        else
        {
          v33 = 0;
          v34 = 0;
        }
        if ( v34 >= *(_DWORD *)(a1 + 56) )
        {
          v90 = 0;
          goto LABEL_113;
        }
        v35 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v33);
        v36 = *(_DWORD *)(v35 + 16);
        v90 = v35;
        if ( v21 < v36 )
        {
          v22 = v102 & 1;
          if ( (v102 & 1) != 0 )
          {
            v37 = (__int64 *)&v103;
            v38 = 7;
          }
          else
          {
            v41 = v104;
            v37 = v103;
            if ( !v104 )
            {
              v42 = v102;
              ++v101;
              v91 = 0;
              v43 = ((unsigned int)v102 >> 1) + 1;
              goto LABEL_39;
            }
            v38 = v104 - 1;
          }
          v39 = v38 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v24 = (__int64)&v37[v39];
          v23 = *(_QWORD *)v24;
          if ( v35 == *(_QWORD *)v24 )
            goto LABEL_28;
          v58 = 1;
          v59 = 0;
          while ( v23 != -4096 )
          {
            if ( v23 != -8192 || v59 )
              v24 = (__int64)v59;
            v39 = v38 & (v58 + v39);
            v23 = v37[v39];
            if ( v35 == v23 )
              goto LABEL_28;
            ++v58;
            v59 = (__int64 *)v24;
            v24 = (__int64)&v37[v39];
          }
          v42 = v102;
          if ( !v59 )
            v59 = (__int64 *)v24;
          ++v101;
          v91 = v59;
          v43 = ((unsigned int)v102 >> 1) + 1;
          if ( (_BYTE)v22 )
          {
            v23 = 24;
            v41 = 8;
            if ( 4 * v43 >= 0x18 )
              goto LABEL_64;
            goto LABEL_40;
          }
          v41 = v104;
LABEL_39:
          v23 = 3 * v41;
          if ( 4 * v43 >= (unsigned int)v23 )
          {
LABEL_64:
            v41 *= 2;
LABEL_65:
            sub_2E72D00((__int64)&v101, v41);
            sub_2E6EF00((__int64)&v101, &v90, &v91);
            v35 = v90;
            v42 = v102;
LABEL_41:
            LODWORD(v102) = (2 * (v42 >> 1) + 2) | v42 & 1;
            if ( *v91 != -4096 )
              --HIDWORD(v102);
            v44 = v88 < v36;
            *v91 = v35;
            v45 = v90;
            if ( v44 )
            {
              v46 = (unsigned int)v93;
              v47 = (unsigned int)v93 + 1LL;
              if ( v47 > HIDWORD(v93) )
              {
                sub_C8D5F0((__int64)&v92, v94, v47, 8u, v23, v24);
                v46 = (unsigned int)v93;
              }
              *(_QWORD *)&v92[8 * v46] = v45;
              v48 = (unsigned int)v109;
              v22 = HIDWORD(v109);
              LODWORD(v93) = v93 + 1;
              v49 = (unsigned int)v109 + 1LL;
              v50 = v90;
              if ( v49 > HIDWORD(v109) )
              {
                sub_C8D5F0((__int64)&v108, v110, v49, 8u, v23, v24);
                v48 = (unsigned int)v109;
              }
              *(_QWORD *)&v108[8 * v48] = v50;
              LODWORD(v109) = v109 + 1;
            }
            else
            {
              v51 = (unsigned int)v99;
              v52 = (unsigned int)v99 + 1LL;
              if ( v52 > HIDWORD(v99) )
              {
                sub_C8D5F0((__int64)&v98, v100, v52, 8u, v23, v24);
                v51 = (unsigned int)v99;
              }
              *(_QWORD *)&v98[8 * v51] = v45;
              v53 = (__int64)v98;
              LODWORD(v99) = v99 + 1;
              v54 = 8LL * (unsigned int)v99;
              v23 = *(_QWORD *)&v98[v54 - 8];
              v55 = (v54 >> 3) - 1;
              v56 = ((v54 >> 3) - 2) / 2;
              if ( v55 > 0 )
              {
                while ( 1 )
                {
                  v57 = v53 + 8 * v56;
                  v22 = v53 + 8 * v55;
                  if ( *(_DWORD *)(*(_QWORD *)v57 + 16LL) >= *(_DWORD *)(v23 + 16) )
                  {
                    *(_QWORD *)v22 = v23;
                    goto LABEL_28;
                  }
                  *(_QWORD *)v22 = *(_QWORD *)v57;
                  v55 = v56;
                  if ( v56 <= 0 )
                    break;
                  v56 = (v56 - 1) / 2;
                }
                v22 = v53 + 8 * v56;
                *(_QWORD *)v57 = v23;
              }
              else
              {
                v22 = (__int64)&v98[v54 - 8];
                *(_QWORD *)v22 = v23;
              }
            }
            goto LABEL_28;
          }
LABEL_40:
          if ( v41 - HIDWORD(v102) - v43 > v41 >> 3 )
            goto LABEL_41;
          goto LABEL_65;
        }
LABEL_28:
        ++v32;
      }
      while ( v31 != v32 );
      v30 = v95;
LABEL_34:
      if ( v30 != (__int64 *)v97 )
        _libc_free((unsigned __int64)v30);
      if ( (_DWORD)v93 )
      {
        v29 = (unsigned int)v93;
        v28 = *(_QWORD *)&v92[8 * (unsigned int)v93 - 8];
        LODWORD(v93) = v93 - 1;
        continue;
      }
      break;
    }
    v25 = (unsigned int)v99;
  }
  while ( (_DWORD)v99 );
LABEL_67:
  v60 = v105;
  v61 = &v105[(unsigned int)v106];
  if ( v105 != v61 )
  {
    do
    {
      v62 = *v60++;
      sub_2E6CC90(v62, v86);
    }
    while ( v61 != v60 );
  }
  v63 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
  if ( v63 != sub_2EB5800(*(__int64 **)a1, v63, a2, v22, v23) )
  {
    sub_2EB9A60(&v95, a1, a2, v64, v65, v66);
    if ( !(unsigned __int8)sub_2EB4750(a1, (__int64)&v95, v67, v68, v69, v70) )
      sub_2EBA1B0(a1, a2);
    if ( v95 != (__int64 *)v97 )
      _libc_free((unsigned __int64)v95);
  }
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
  if ( v105 != (__int64 *)v107 )
    _libc_free((unsigned __int64)v105);
  if ( (v102 & 1) == 0 )
    sub_C7D6A0((__int64)v103, 8LL * v104, 8);
  if ( v98 != v100 )
    _libc_free((unsigned __int64)v98);
}
