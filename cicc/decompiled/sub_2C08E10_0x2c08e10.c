// Function: sub_2C08E10
// Address: 0x2c08e10
//
__int64 __fastcall sub_2C08E10(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  int v12; // eax
  size_t v13; // r14
  size_t v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r9
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rdi
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // r8d
  __int64 *v25; // rax
  __int64 v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // rax
  int v29; // ecx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r8
  _QWORD **v36; // rdi
  _QWORD *v37; // rdx
  _QWORD *v38; // rax
  unsigned int v39; // ecx
  _QWORD *v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  __int64 v43; // rcx
  int v44; // esi
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // r13
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // r14
  _BYTE *v55; // rsi
  size_t v56; // rdx
  __int64 v57; // rax
  void **v58; // rcx
  int v59; // edx
  _QWORD *v60; // rdi
  int v61; // edx
  __int64 v62; // rsi
  int v63; // r15d
  _QWORD *v64; // r9
  unsigned int v65; // r8d
  _QWORD *v66; // rax
  _QWORD *v67; // r11
  __int64 v68; // rax
  int v69; // r10d
  void ***v70; // r9
  unsigned int v71; // edi
  void ***v72; // rax
  void **v73; // r8
  int v74; // r8d
  int v75; // eax
  int v76; // r8d
  int v77; // r15d
  __int64 *v78; // rcx
  int v79; // eax
  int v80; // edi
  int v81; // eax
  int v82; // esi
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // r8
  int v86; // r10d
  __int64 *v87; // r9
  _QWORD *v88; // rax
  void *v89; // rdx
  int v90; // edx
  void **v91; // rdx
  int v92; // eax
  int v93; // eax
  __int64 v94; // r8
  int v95; // r10d
  __int64 v96; // rdx
  __int64 v97; // rsi
  int v98; // eax
  int v99; // r9d
  _QWORD *v100; // rdi
  int v101; // eax
  unsigned int v102; // [rsp+4h] [rbp-9Ch]
  const char *src; // [rsp+8h] [rbp-98h]
  void **v104; // [rsp+18h] [rbp-88h] BYREF
  __int64 v105[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v106; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v107; // [rsp+40h] [rbp-60h] BYREF
  size_t v108; // [rsp+48h] [rbp-58h]
  _QWORD v109[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v110; // [rsp+60h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 64);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      v10 = v8[1];
      if ( v10 )
        return v10;
    }
    else
    {
      v12 = 1;
      while ( v9 != -4096 )
      {
        v74 = v12 + 1;
        v7 = v6 & (v12 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v12 = v74;
      }
    }
  }
  if ( !*(_QWORD *)a1 || (v13 = 11, src = "vector.body", **(_QWORD **)(*(_QWORD *)a1 + 32LL) != a2) )
  {
    src = sub_BD5D20(a2);
    v13 = v14;
  }
  v108 = v13;
  v15 = *(_QWORD *)(a1 + 16);
  v110 = 261;
  v107 = src;
  v10 = sub_22077B0(0x80u);
  if ( v10 )
  {
    sub_CA0F50(v105, (void **)&v107);
    v18 = (_BYTE *)v105[0];
    *(_BYTE *)(v10 + 8) = 1;
    v19 = v105[1];
    *(_QWORD *)v10 = &unk_4A23970;
    *(_QWORD *)(v10 + 16) = v10 + 32;
    sub_2C084F0((__int64 *)(v10 + 16), v18, (__int64)&v18[v19]);
    v20 = (__int64 *)v105[0];
    *(_QWORD *)(v10 + 56) = v10 + 72;
    *(_QWORD *)(v10 + 64) = 0x100000000LL;
    *(_QWORD *)(v10 + 88) = 0x100000000LL;
    *(_QWORD *)(v10 + 48) = 0;
    *(_QWORD *)(v10 + 80) = v10 + 96;
    *(_QWORD *)(v10 + 104) = 0;
    if ( v20 != &v106 )
      j_j___libc_free_0((unsigned __int64)v20);
    *(_QWORD *)v10 = &unk_4A23A00;
    *(_QWORD *)(v10 + 120) = v10 + 112;
    *(_QWORD *)(v10 + 112) = (v10 + 112) | 4;
  }
  v21 = *(unsigned int *)(v15 + 600);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 604) )
  {
    sub_C8D5F0(v15 + 592, (const void *)(v15 + 608), v21 + 1, 8u, v16, v17);
    v21 = *(unsigned int *)(v15 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(v15 + 592) + 8 * v21) = v10;
  ++*(_DWORD *)(v15 + 600);
  v22 = *(_DWORD *)(a1 + 64);
  if ( !v22 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_77;
  }
  v23 = *(_QWORD *)(a1 + 48);
  v24 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( *v25 == a2 )
  {
LABEL_18:
    v27 = v25 + 1;
    goto LABEL_19;
  }
  v77 = 1;
  v78 = 0;
  while ( v26 != -4096 )
  {
    if ( v26 == -8192 && !v78 )
      v78 = v25;
    v101 = v77++;
    v24 = (v22 - 1) & (v101 + v24);
    v25 = (__int64 *)(v23 + 16LL * v24);
    v26 = *v25;
    if ( *v25 == a2 )
      goto LABEL_18;
  }
  if ( !v78 )
    v78 = v25;
  v79 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v80 = v79 + 1;
  if ( 4 * (v79 + 1) >= 3 * v22 )
  {
LABEL_77:
    sub_2C088B0(a1 + 40, 2 * v22);
    v81 = *(_DWORD *)(a1 + 64);
    if ( v81 )
    {
      v82 = v81 - 1;
      v83 = *(_QWORD *)(a1 + 48);
      LODWORD(v84) = (v81 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v80 = *(_DWORD *)(a1 + 56) + 1;
      v78 = (__int64 *)(v83 + 16LL * (unsigned int)v84);
      v85 = *v78;
      if ( *v78 == a2 )
        goto LABEL_72;
      v86 = 1;
      v87 = 0;
      while ( v85 != -4096 )
      {
        if ( v85 == -8192 && !v87 )
          v87 = v78;
        v84 = v82 & (unsigned int)(v84 + v86);
        v78 = (__int64 *)(v83 + 16 * v84);
        v85 = *v78;
        if ( *v78 == a2 )
          goto LABEL_72;
        ++v86;
      }
LABEL_81:
      if ( v87 )
        v78 = v87;
      goto LABEL_72;
    }
LABEL_133:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
  if ( v22 - *(_DWORD *)(a1 + 60) - v80 <= v22 >> 3 )
  {
    v102 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    sub_2C088B0(a1 + 40, v22);
    v92 = *(_DWORD *)(a1 + 64);
    if ( v92 )
    {
      v93 = v92 - 1;
      v94 = *(_QWORD *)(a1 + 48);
      v87 = 0;
      v95 = 1;
      LODWORD(v96) = v93 & v102;
      v80 = *(_DWORD *)(a1 + 56) + 1;
      v78 = (__int64 *)(v94 + 16LL * (v93 & v102));
      v97 = *v78;
      if ( *v78 == a2 )
        goto LABEL_72;
      while ( v97 != -4096 )
      {
        if ( !v87 && v97 == -8192 )
          v87 = v78;
        v96 = v93 & (unsigned int)(v96 + v95);
        v78 = (__int64 *)(v94 + 16 * v96);
        v97 = *v78;
        if ( *v78 == a2 )
          goto LABEL_72;
        ++v95;
      }
      goto LABEL_81;
    }
    goto LABEL_133;
  }
LABEL_72:
  *(_DWORD *)(a1 + 56) = v80;
  if ( *v78 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v78 = a2;
  v27 = v78 + 1;
  v78[1] = 0;
LABEL_19:
  *v27 = v10;
  v28 = *(_QWORD *)(a1 + 8);
  v29 = *(_DWORD *)(v28 + 24);
  v30 = *(_QWORD *)(v28 + 8);
  if ( !v29 )
    return v10;
  v31 = v29 - 1;
  v32 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v33 = (__int64 *)(v30 + 16LL * v32);
  v34 = *v33;
  if ( *v33 == a2 )
  {
LABEL_21:
    v35 = v33[1];
    v104 = (void **)v35;
    if ( v35 )
    {
      v36 = *(_QWORD ***)a1;
      if ( *(_QWORD *)a1 != v35 )
      {
        v37 = *(_QWORD **)v35;
        if ( *(_QWORD *)v35 )
        {
          v38 = *(_QWORD **)v35;
          v39 = 1;
          do
          {
            v38 = (_QWORD *)*v38;
            ++v39;
          }
          while ( v38 );
          v40 = *v36;
          if ( !*v36 )
          {
            v41 = 1;
            goto LABEL_29;
          }
        }
        else
        {
          v40 = *v36;
          if ( !*v36 )
            return v10;
          v39 = 1;
        }
        v41 = 1;
        do
        {
          v40 = (_QWORD *)*v40;
          ++v41;
        }
        while ( v40 );
LABEL_29:
        if ( v39 < v41 )
          return v10;
        while ( 1 )
        {
          if ( !v37 )
            return v10;
          if ( v36 == v37 )
            break;
          v37 = (_QWORD *)*v37;
        }
        v42 = *(_DWORD *)(a1 + 208);
        v43 = *(_QWORD *)(a1 + 192);
        if ( v42 )
        {
          v44 = v42 - 1;
          v45 = (v42 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v46 = (__int64 *)(v43 + 16LL * v45);
          v47 = *v46;
          if ( v35 == *v46 )
          {
LABEL_37:
            v48 = v46[1];
            goto LABEL_38;
          }
          v98 = 1;
          while ( v47 != -4096 )
          {
            v99 = v98 + 1;
            v45 = v44 & (v98 + v45);
            v46 = (__int64 *)(v43 + 16LL * v45);
            v47 = *v46;
            if ( v35 == *v46 )
              goto LABEL_37;
            v98 = v99;
          }
        }
        v48 = 0;
LABEL_38:
        if ( **(_QWORD **)(v35 + 32) != a2 )
        {
          *(_QWORD *)(v10 + 48) = v48;
          return v10;
        }
        v49 = *(_QWORD *)(a1 + 16);
        if ( !src )
        {
          v108 = 0;
          v107 = v109;
          LOBYTE(v109[0]) = 0;
LABEL_46:
          v51 = sub_22077B0(0x88u);
          v54 = v51;
          if ( v51 )
          {
            v55 = v107;
            *(_BYTE *)(v51 + 8) = 0;
            v56 = v108;
            *(_QWORD *)v51 = &unk_4A23970;
            *(_QWORD *)(v51 + 16) = v51 + 32;
            sub_2C084F0((__int64 *)(v51 + 16), v55, (__int64)&v55[v56]);
            *(_QWORD *)(v54 + 48) = 0;
            *(_QWORD *)(v54 + 56) = v54 + 72;
            *(_QWORD *)(v54 + 64) = 0x100000000LL;
            *(_QWORD *)(v54 + 88) = 0x100000000LL;
            *(_QWORD *)(v54 + 80) = v54 + 96;
            *(_QWORD *)(v54 + 104) = 0;
            *(_QWORD *)v54 = &unk_4A23A38;
            *(_QWORD *)(v54 + 112) = 0;
            *(_QWORD *)(v54 + 120) = 0;
            *(_BYTE *)(v54 + 128) = 0;
          }
          v57 = *(unsigned int *)(v49 + 600);
          if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(v49 + 604) )
          {
            sub_C8D5F0(v49 + 592, (const void *)(v49 + 608), v57 + 1, 8u, v52, v53);
            v57 = *(unsigned int *)(v49 + 600);
          }
          *(_QWORD *)(*(_QWORD *)(v49 + 592) + 8 * v57) = v54;
          ++*(_DWORD *)(v49 + 600);
          if ( v107 != v109 )
            j_j___libc_free_0((unsigned __int64)v107);
          v58 = v104;
          v59 = *(_DWORD *)(a1 + 208);
          v60 = *v104;
          v107 = *v104;
          if ( v59 )
          {
            v61 = v59 - 1;
            v62 = *(_QWORD *)(a1 + 192);
            v63 = 1;
            v64 = 0;
            v65 = v61 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v66 = (_QWORD *)(v62 + 16LL * v65);
            v67 = (_QWORD *)*v66;
            if ( v60 == (_QWORD *)*v66 )
            {
LABEL_54:
              v68 = v66[1];
              *(_QWORD *)(v54 + 112) = v10;
              *(_QWORD *)(v10 + 48) = v54;
              *(_QWORD *)(v54 + 48) = v68;
              goto LABEL_55;
            }
            while ( v67 != (_QWORD *)-4096LL )
            {
              if ( !v64 && v67 == (_QWORD *)-8192LL )
                v64 = v66;
              v65 = v61 & (v63 + v65);
              v66 = (_QWORD *)(v62 + 16LL * v65);
              v67 = (_QWORD *)*v66;
              if ( v60 == (_QWORD *)*v66 )
                goto LABEL_54;
              ++v63;
            }
            if ( !v64 )
              v64 = v66;
          }
          else
          {
            v64 = 0;
          }
          v88 = sub_2C08C70(a1 + 184, &v107, v64);
          v89 = v107;
          v70 = 0;
          v88[1] = 0;
          *v88 = v89;
          v90 = *(_DWORD *)(a1 + 208);
          *(_QWORD *)(v54 + 48) = 0;
          v62 = *(_QWORD *)(a1 + 192);
          *(_QWORD *)(v54 + 112) = v10;
          *(_QWORD *)(v10 + 48) = v54;
          if ( !v90 )
            goto LABEL_86;
          v58 = v104;
          v61 = v90 - 1;
LABEL_55:
          v69 = 1;
          v70 = 0;
          v71 = v61 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
          v72 = (void ***)(v62 + 16LL * v71);
          v73 = *v72;
          if ( *v72 == v58 )
          {
LABEL_56:
            v72[1] = (void **)v54;
            return v10;
          }
          while ( v73 != (void **)-4096LL )
          {
            if ( !v70 && v73 == (void **)-8192LL )
              v70 = v72;
            v71 = v61 & (v69 + v71);
            v72 = (void ***)(v62 + 16LL * v71);
            v73 = *v72;
            if ( v58 == *v72 )
              goto LABEL_56;
            ++v69;
          }
          if ( !v70 )
            v70 = v72;
LABEL_86:
          v72 = (void ***)sub_2C08C70(a1 + 184, &v104, v70);
          v91 = v104;
          v72[1] = 0;
          *v72 = v91;
          goto LABEL_56;
        }
        v105[0] = v13;
        v107 = v109;
        if ( v13 > 0xF )
        {
          v107 = (_QWORD *)sub_22409D0((__int64)&v107, (unsigned __int64 *)v105, 0);
          v100 = v107;
          v109[0] = v105[0];
        }
        else
        {
          if ( v13 == 1 )
          {
            LOBYTE(v109[0]) = *src;
            v50 = v109;
LABEL_45:
            v108 = v13;
            *((_BYTE *)v50 + v13) = 0;
            goto LABEL_46;
          }
          if ( !v13 )
          {
            v50 = v109;
            goto LABEL_45;
          }
          v100 = v109;
        }
        memcpy(v100, src, v13);
        v13 = v105[0];
        v50 = v107;
        goto LABEL_45;
      }
    }
  }
  else
  {
    v75 = 1;
    while ( v34 != -4096 )
    {
      v76 = v75 + 1;
      v32 = v31 & (v75 + v32);
      v33 = (__int64 *)(v30 + 16LL * v32);
      v34 = *v33;
      if ( *v33 == a2 )
        goto LABEL_21;
      v75 = v76;
    }
  }
  return v10;
}
