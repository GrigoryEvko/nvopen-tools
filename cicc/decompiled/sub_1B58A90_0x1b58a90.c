// Function: sub_1B58A90
// Address: 0x1b58a90
//
__int64 __fastcall sub_1B58A90(__int64 a1, __int64 *a2, _BYTE *a3, __int64 a4)
{
  __int64 v4; // r12
  char v5; // al
  unsigned int v6; // edx
  char v7; // di
  _BYTE **v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  char v16; // al
  const char *v17; // rdi
  _BYTE *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rbx
  __int64 v29; // r13
  _BYTE *v30; // r8
  int v31; // r9d
  unsigned int v32; // edx
  __int64 v33; // rcx
  __int64 *v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdi
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  _BYTE *v45; // r12
  __int64 v46; // rbx
  _BYTE *v47; // r13
  __int64 v48; // rcx
  __int64 v49; // rcx
  char **v50; // rsi
  __int64 v51; // rdi
  _BYTE *v52; // rbx
  _BYTE *v53; // r14
  unsigned __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  __int64 v58; // r14
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rax
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rbx
  unsigned int v67; // esi
  int v68; // eax
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r14
  _QWORD *v72; // rax
  _QWORD *v73; // r13
  __int64 v74; // rax
  unsigned __int64 *v75; // r14
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rsi
  __int64 v79; // rsi
  __int64 v80; // rdx
  unsigned __int8 *v81; // rsi
  __int64 v82; // r14
  __int64 v83; // r13
  __int64 v84; // rax
  __int64 v85; // rdi
  int v86; // [rsp-10h] [rbp-210h]
  int v87; // [rsp-8h] [rbp-208h]
  __int64 v88; // [rsp+18h] [rbp-1E8h]
  __int64 v89; // [rsp+18h] [rbp-1E8h]
  unsigned int v90; // [rsp+20h] [rbp-1E0h]
  __int64 v91; // [rsp+20h] [rbp-1E0h]
  __int64 v92; // [rsp+20h] [rbp-1E0h]
  __int64 v93; // [rsp+20h] [rbp-1E0h]
  int v94; // [rsp+28h] [rbp-1D8h]
  _BYTE *v95; // [rsp+30h] [rbp-1D0h]
  __int64 v97; // [rsp+50h] [rbp-1B0h]
  __int64 v100; // [rsp+88h] [rbp-178h] BYREF
  char *v101[2]; // [rsp+90h] [rbp-170h] BYREF
  _QWORD v102[4]; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v103; // [rsp+C0h] [rbp-140h] BYREF
  char *v104; // [rsp+C8h] [rbp-138h] BYREF
  __int64 v105; // [rsp+D0h] [rbp-130h]
  _BYTE v106[40]; // [rsp+D8h] [rbp-128h] BYREF
  const char *v107; // [rsp+100h] [rbp-100h] BYREF
  __int64 v108; // [rsp+108h] [rbp-F8h]
  _WORD v109[32]; // [rsp+110h] [rbp-F0h] BYREF
  _BYTE *v110; // [rsp+150h] [rbp-B0h]
  __int64 v111; // [rsp+158h] [rbp-A8h]
  _BYTE v112[160]; // [rsp+160h] [rbp-A0h] BYREF

  v4 = a1;
  v5 = *(_BYTE *)(a1 + 23);
  v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v7 = v5 & 0x40;
  if ( (v5 & 0x40) != 0 )
    v8 = *(_BYTE ***)(v4 - 8);
  else
    v8 = (_BYTE **)(v4 - 24LL * v6);
  v9 = v6 >> 1;
  v10 = 0;
  v95 = *v8;
  v110 = v112;
  v100 = 0;
  v111 = 0x200000000LL;
  v97 = v9 - 1;
  if ( v9 != 1 )
  {
    v11 = 0;
    while ( 1 )
    {
      v12 = 2 * ++v11;
      if ( (v5 & 0x40) != 0 )
        v13 = *(_QWORD *)(v4 - 8);
      else
        v13 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
      v14 = *(_QWORD *)(v13 + 24LL * v12);
      v107 = (const char *)v109;
      v108 = 0x400000000LL;
      v15 = 24;
      if ( (_DWORD)v11 != -1 )
        v15 = 24LL * (v12 + 1);
      v16 = sub_1B57F80(v4, v14, *(_QWORD *)(v13 + v15), &v100, (__int64)&v107, a3, a4);
      v17 = v107;
      if ( !v16 || (unsigned int)v108 > 1 )
        goto LABEL_44;
      v18 = v110;
      v19 = *((_QWORD *)v107 + 1);
      v20 = (__int64)&v110[56 * (unsigned int)v111];
      if ( v110 == (_BYTE *)v20 )
      {
LABEL_34:
        v102[0] = v14;
        v103 = v19;
        v104 = v106;
        v105 = 0x400000000LL;
        v101[0] = (char *)v102;
        v101[1] = (char *)0x400000001LL;
        sub_1B42E40((__int64)&v104, v101, v20, 0x400000001LL, v86, v87);
        v32 = v111;
        if ( (unsigned int)v111 >= HIDWORD(v111) )
        {
          v90 = v111;
          v39 = (((((unsigned __int64)HIDWORD(v111) + 2) >> 1) | (HIDWORD(v111) + 2LL)) >> 2)
              | (((unsigned __int64)HIDWORD(v111) + 2) >> 1)
              | (HIDWORD(v111) + 2LL);
          v40 = (((v39 >> 4) | v39) >> 8) | (v39 >> 4) | v39;
          v41 = (v40 | (v40 >> 16) | HIDWORD(v40)) + 1;
          v42 = 0xFFFFFFFFLL;
          if ( v41 <= 0xFFFFFFFF )
            v42 = v41;
          v94 = v42;
          v43 = malloc(56 * v42);
          v44 = v90;
          v33 = v43;
          if ( !v43 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v44 = (unsigned int)v111;
            v33 = 0;
          }
          v30 = &v110[56 * v44];
          if ( v110 != v30 )
          {
            v88 = v4;
            v45 = &v110[56 * v44];
            v46 = v33;
            v47 = v110;
            v91 = v33;
            do
            {
              while ( 1 )
              {
                if ( v46 )
                {
                  v48 = *(_QWORD *)v47;
                  *(_DWORD *)(v46 + 16) = 0;
                  *(_DWORD *)(v46 + 20) = 4;
                  *(_QWORD *)v46 = v48;
                  v49 = v46 + 24;
                  *(_QWORD *)(v46 + 8) = v46 + 24;
                  if ( *((_DWORD *)v47 + 4) )
                    break;
                }
                v47 += 56;
                v46 += 56;
                if ( v45 == v47 )
                  goto LABEL_68;
              }
              v50 = (char **)(v47 + 8);
              v51 = v46 + 8;
              v47 += 56;
              v46 += 56;
              sub_1B42E40(v51, v50, v44, v49, (int)v30, v31);
            }
            while ( v45 != v47 );
LABEL_68:
            v30 = v110;
            v33 = v91;
            v4 = v88;
            if ( v110 != &v110[56 * (unsigned int)v111] )
            {
              v89 = v91;
              v52 = &v110[56 * (unsigned int)v111];
              v92 = v11;
              v53 = v110;
              do
              {
                v52 -= 56;
                v54 = *((_QWORD *)v52 + 1);
                if ( (_BYTE *)v54 != v52 + 24 )
                  _libc_free(v54);
              }
              while ( v52 != v53 );
              v11 = v92;
              v33 = v89;
              v30 = v110;
            }
          }
          if ( v30 != v112 )
          {
            v93 = v33;
            _libc_free((unsigned __int64)v30);
            v33 = v93;
          }
          v110 = (_BYTE *)v33;
          v32 = v111;
          HIDWORD(v111) = v94;
        }
        else
        {
          v33 = (__int64)v110;
        }
        v34 = (__int64 *)(v33 + 56LL * v32);
        if ( v34 )
        {
          v35 = v103;
          v34[2] = 0x400000000LL;
          *v34 = v35;
          v34[1] = (__int64)(v34 + 3);
          if ( (_DWORD)v105 )
            sub_1B42E40((__int64)(v34 + 1), &v104, (__int64)(v34 + 3), v33, (int)v30, v31);
          v32 = v111;
        }
        LODWORD(v111) = v32 + 1;
        if ( v104 != v106 )
          _libc_free((unsigned __int64)v104);
        if ( (_QWORD *)v101[0] != v102 )
        {
          _libc_free((unsigned __int64)v101[0]);
          v17 = v107;
          if ( (unsigned int)v111 > 2 )
            goto LABEL_44;
          goto LABEL_19;
        }
      }
      else
      {
        while ( v19 != *(_QWORD *)v18 )
        {
          v18 += 56;
          if ( (_BYTE *)v20 == v18 )
            goto LABEL_34;
        }
        v21 = *((unsigned int *)v18 + 4);
        if ( (unsigned int)v21 >= *((_DWORD *)v18 + 5) )
        {
          sub_16CD150((__int64)(v18 + 8), v18 + 24, 0, 8, v86, v87);
          v21 = *((unsigned int *)v18 + 4);
        }
        *(_QWORD *)(*((_QWORD *)v18 + 1) + 8 * v21) = v14;
        v22 = *((_DWORD *)v18 + 4) + 1;
        *((_DWORD *)v18 + 4) = v22;
        if ( v22 > 1 )
        {
          v17 = v107;
          if ( v107 != (const char *)v109 )
            goto LABEL_45;
          goto LABEL_46;
        }
      }
      v17 = v107;
      if ( (unsigned int)v111 > 2 )
        goto LABEL_44;
LABEL_19:
      if ( v10 )
      {
        if ( *(_QWORD *)v17 != v10 )
          goto LABEL_44;
      }
      else
      {
        v10 = *(_QWORD *)v17;
      }
      if ( v17 != (const char *)v109 )
        _libc_free((unsigned __int64)v17);
      v5 = *(_BYTE *)(v4 + 23);
      if ( v97 == v11 )
      {
        v7 = v5 & 0x40;
        break;
      }
    }
  }
  v107 = (const char *)v109;
  v108 = 0x100000000LL;
  if ( v7 )
    v23 = *(_QWORD *)(v4 - 8);
  else
    v23 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
  v24 = *(_QWORD *)(v23 + 24);
  sub_1B57F80(v4, 0, v24, &v100, (__int64)&v107, a3, a4);
  if ( (_DWORD)v108 != 1 || (v17 = v107, (v25 = *((_QWORD *)v107 + 1)) == 0) )
  {
    v25 = 0;
    v26 = sub_157ED60(v24);
    v17 = v107;
    if ( *(_BYTE *)(v26 + 16) != 31 )
    {
LABEL_44:
      if ( v17 != (const char *)v109 )
LABEL_45:
        _libc_free((unsigned __int64)v17);
LABEL_46:
      v28 = v110;
      goto LABEL_47;
    }
  }
  if ( v17 != (const char *)v109 )
    _libc_free((unsigned __int64)v17);
  v27 = (unsigned int)v111;
  if ( (_DWORD)v111 != 2 )
  {
    v28 = v110;
    LODWORD(v29) = 0;
    goto LABEL_48;
  }
  a2[1] = *(_QWORD *)(v4 + 40);
  a2[2] = v4 + 24;
  v55 = *(_QWORD *)(v4 + 48);
  v107 = (const char *)v55;
  if ( v55 )
  {
    sub_1623A60((__int64)&v107, v55, 2);
    v56 = *a2;
    if ( !*a2 )
      goto LABEL_82;
  }
  else
  {
    v56 = *a2;
    if ( !*a2 )
      goto LABEL_84;
  }
  sub_161E7C0((__int64)a2, v56);
LABEL_82:
  v57 = (unsigned __int8 *)v107;
  *a2 = (__int64)v107;
  if ( v57 )
    sub_1623210((__int64)&v107, v57, (__int64)a2);
LABEL_84:
  v28 = v110;
  if ( *((_DWORD *)v110 + 4) != 1 || *((_DWORD *)v110 + 18) != 1 )
  {
LABEL_47:
    v27 = (unsigned int)v111;
    LODWORD(v29) = 0;
    goto LABEL_48;
  }
  v58 = **((_QWORD **)v110 + 1);
  if ( v25 )
  {
    v59 = **((_QWORD **)v110 + 8);
    v107 = "switch.selectcmp";
    v109[0] = 259;
    v60 = sub_12AA0C0(a2, 0x20u, v95, v59, (__int64)&v107);
    v107 = "switch.select";
    v109[0] = 259;
    v61 = sub_156B790(a2, v60, *((_QWORD *)v110 + 7), v25, (__int64)&v107, 0);
  }
  else
  {
    v61 = *((_QWORD *)v110 + 7);
  }
  v107 = "switch.selectcmp";
  v109[0] = 259;
  v62 = sub_12AA0C0(a2, 0x20u, v95, v58, (__int64)&v107);
  v107 = "switch.select";
  v109[0] = 259;
  v29 = sub_156B790(a2, v62, *(_QWORD *)v110, v61, (__int64)&v107, 0);
  if ( v29 )
  {
    v66 = *(_QWORD *)(v4 + 40);
LABEL_96:
    while ( 1 )
    {
      v67 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      if ( !v67 )
        break;
      v64 = v10 - 24LL * v67;
      v68 = 0;
      v65 = 24LL * *(unsigned int *)(v10 + 56);
      v69 = v65 + 8;
      while ( 1 )
      {
        v63 = v10 - 24LL * v67;
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
          v63 = *(_QWORD *)(v10 - 8);
        if ( v66 == *(_QWORD *)(v63 + v69) )
          break;
        ++v68;
        v69 += 8;
        if ( v67 == v68 )
          goto LABEL_108;
      }
      if ( v68 < 0 )
        break;
      v70 = 0;
      do
      {
        if ( v66 == *(_QWORD *)(v63 + v65 + 8 * v70 + 8) )
        {
          sub_15F5350(v10, v70, 1);
          goto LABEL_96;
        }
        ++v70;
      }
      while ( v67 != (_DWORD)v70 );
      sub_15F5350(v10, 0xFFFFFFFF, 1);
    }
LABEL_108:
    sub_1704F80(v10, v29, v66, v63, v64, v65);
    v71 = *(_QWORD *)(v10 + 40);
    v109[0] = 257;
    v72 = sub_1648A60(56, 1u);
    v73 = v72;
    if ( v72 )
      sub_15F8320((__int64)v72, v71, 0);
    v74 = a2[1];
    if ( v74 )
    {
      v75 = (unsigned __int64 *)a2[2];
      sub_157E9D0(v74 + 40, (__int64)v73);
      v76 = v73[3];
      v77 = *v75;
      v73[4] = v75;
      v77 &= 0xFFFFFFFFFFFFFFF8LL;
      v73[3] = v77 | v76 & 7;
      *(_QWORD *)(v77 + 8) = v73 + 3;
      *v75 = *v75 & 7 | (unsigned __int64)(v73 + 3);
    }
    sub_164B780((__int64)v73, (__int64 *)&v107);
    v78 = *a2;
    if ( *a2 )
    {
      v103 = *a2;
      sub_1623A60((__int64)&v103, v78, 2);
      v79 = v73[6];
      v80 = (__int64)(v73 + 6);
      if ( v79 )
      {
        sub_161E7C0((__int64)(v73 + 6), v79);
        v80 = (__int64)(v73 + 6);
      }
      v81 = (unsigned __int8 *)v103;
      v73[6] = v103;
      if ( v81 )
        sub_1623210((__int64)&v103, v81, v80);
    }
    if ( (*(_DWORD *)(v4 + 20) & 0xFFFFFFFu) >> 1 )
    {
      v82 = 24;
      v83 = 48LL * (((*(_DWORD *)(v4 + 20) & 0xFFFFFFFu) >> 1) - 1) + 72;
      do
      {
        if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
          v84 = *(_QWORD *)(v4 - 8);
        else
          v84 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
        v85 = *(_QWORD *)(v84 + v82);
        if ( *(_QWORD *)(v10 + 40) != v85 )
          sub_157F2D0(v85, v66, 0);
        v82 += 48;
      }
      while ( v83 != v82 );
    }
    LODWORD(v29) = 1;
    sub_15F20C0((_QWORD *)v4);
    v28 = v110;
    v27 = (unsigned int)v111;
  }
  else
  {
    v28 = v110;
    v27 = (unsigned int)v111;
  }
LABEL_48:
  v36 = (unsigned __int64)&v28[56 * v27];
  if ( v28 != (_BYTE *)v36 )
  {
    do
    {
      v36 -= 56LL;
      v37 = *(_QWORD *)(v36 + 8);
      if ( v37 != v36 + 24 )
        _libc_free(v37);
    }
    while ( v28 != (_BYTE *)v36 );
    v36 = (unsigned __int64)v110;
  }
  if ( (_BYTE *)v36 != v112 )
    _libc_free(v36);
  return (unsigned int)v29;
}
