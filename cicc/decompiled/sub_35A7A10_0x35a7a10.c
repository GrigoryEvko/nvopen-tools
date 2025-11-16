// Function: sub_35A7A10
// Address: 0x35a7a10
//
__int64 __fastcall sub_35A7A10(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 *v31; // rax
  __int64 v32; // r13
  unsigned int v33; // r15d
  __int64 *v34; // r14
  __int64 v35; // rdi
  __int64 v36; // rbx
  unsigned __int64 v37; // r11
  __int64 v38; // r10
  unsigned __int64 v39; // r13
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned int v42; // ecx
  __int64 *v43; // rdx
  __int64 v44; // r8
  __int64 *v46; // rdi
  __int64 v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // rsi
  __int64 v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // r8d
  __int64 v54; // r9
  unsigned int v55; // ecx
  __int64 *v56; // rdx
  __int64 v57; // rsi
  __int64 *v58; // rax
  int v59; // edx
  int v60; // r9d
  __int64 *v61; // rax
  int v62; // edx
  unsigned __int64 v63; // rax
  __int64 v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // r9
  _QWORD *i; // rdx
  __int64 *v68; // rsi
  __int64 v69; // r11
  int v70; // r8d
  _QWORD *v71; // rcx
  unsigned int v72; // edx
  _QWORD *v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rcx
  unsigned int v76; // esi
  unsigned int v77; // r11d
  unsigned int v78; // esi
  __int64 v79; // r9
  int v80; // r8d
  __int64 *v81; // rdi
  __int64 *v82; // rcx
  unsigned int v83; // r11d
  int v84; // esi
  __int64 v85; // rdi
  _QWORD *v86; // rdx
  __int64 v88; // [rsp+18h] [rbp-B8h]
  __int64 v89; // [rsp+20h] [rbp-B0h]
  __int64 v90; // [rsp+20h] [rbp-B0h]
  __int64 v91; // [rsp+20h] [rbp-B0h]
  __int64 v93; // [rsp+30h] [rbp-A0h]
  __int64 v95; // [rsp+48h] [rbp-88h]
  int v96; // [rsp+48h] [rbp-88h]
  unsigned int v97; // [rsp+48h] [rbp-88h]
  __int64 *v98; // [rsp+48h] [rbp-88h]
  __int64 v99; // [rsp+48h] [rbp-88h]
  __int64 v100; // [rsp+50h] [rbp-80h]
  int v102; // [rsp+5Ch] [rbp-74h]
  __int64 v103; // [rsp+68h] [rbp-68h] BYREF
  __int64 *v104; // [rsp+70h] [rbp-60h]
  __int64 v105; // [rsp+78h] [rbp-58h]
  __int64 v106; // [rsp+80h] [rbp-50h] BYREF
  _QWORD *v107; // [rsp+88h] [rbp-48h]
  __int64 v108; // [rsp+90h] [rbp-40h]
  unsigned int v109; // [rsp+98h] [rbp-38h]

  v5 = a1;
  v6 = a1[7];
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  if ( a2 )
  {
    v102 = 0;
    v93 = v6;
    while ( 1 )
    {
      v7 = v5[6];
      v8 = v5[1];
      LOBYTE(v105) = 0;
      v9 = v93;
      v93 = sub_2E7AAE0(v8, *(_QWORD *)(v7 + 16), (__int64)v104, 0);
      v12 = *(unsigned int *)(a5 + 8);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, (const void *)(a5 + 16), v12 + 1, 8u, v10, v11);
        v12 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v12) = v93;
      ++*(_DWORD *)(a5 + 8);
      v13 = (__int64 *)v5[6];
      sub_2E33BD0(v5[1] + 320, v93);
      v14 = *v13;
      v15 = *(_QWORD *)v93;
      *(_QWORD *)(v93 + 8) = v13;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v93 = v14 | v15 & 7;
      *(_QWORD *)(v14 + 8) = v93;
      *v13 = v93 | *v13 & 7;
      sub_2E340B0(v93, v9, v14, v16, v17, v18);
      sub_2E33F80(v9, v93, -1, v19, v20, v21);
      v22 = v5[5];
      sub_2E34D50(*(_QWORD *)(v22 + 32), (__int64 *)v93, v23, v24, v25, v26);
      v29 = *(unsigned int *)(v22 + 352);
      v30 = *(unsigned int *)(v22 + 192);
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 356) )
      {
        sub_C8D5F0(v22 + 344, (const void *)(v22 + 360), v29 + 1, 8u, v27, v28);
        v29 = *(unsigned int *)(v22 + 352);
      }
      *(_QWORD *)(*(_QWORD *)(v22 + 344) + 8 * v29) = v30;
      ++*(_DWORD *)(v22 + 352);
      if ( v102 >= 0 )
        break;
LABEL_22:
      sub_359C870(v5, v93, v102++, a4, (__int64)&v106);
      if ( a2 == v102 )
      {
        v6 = v93;
        goto LABEL_24;
      }
    }
    v31 = v5;
    v32 = v93;
    v33 = v102;
    v34 = v31;
LABEL_9:
    v35 = v34[6];
    v36 = *(_QWORD *)(v35 + 56);
    v37 = sub_2E313E0(v35);
    v100 = v32 + 48;
    if ( v36 == v37 )
      goto LABEL_20;
    v38 = v32;
    v39 = v37;
    while ( 1 )
    {
      v40 = *(_QWORD *)(*v34 + 72);
      v41 = *(unsigned int *)(*v34 + 88);
      if ( !(_DWORD)v41 )
        goto LABEL_16;
      v42 = (v41 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v43 = (__int64 *)(v40 + 16LL * v42);
      v44 = *v43;
      if ( *v43 != v36 )
      {
        v59 = 1;
        while ( v44 != -4096 )
        {
          v60 = v59 + 1;
          v42 = (v41 - 1) & (v59 + v42);
          v43 = (__int64 *)(v40 + 16LL * v42);
          v44 = *v43;
          if ( *v43 == v36 )
            goto LABEL_14;
          v59 = v60;
        }
LABEL_16:
        if ( !v36 )
          BUG();
        goto LABEL_17;
      }
LABEL_14:
      if ( v43 == (__int64 *)(v40 + 16 * v41) || *((_DWORD *)v43 + 2) != v33 )
        goto LABEL_16;
      if ( *(_WORD *)(v36 + 68) && *(_WORD *)(v36 + 68) != 68 )
        break;
LABEL_17:
      if ( (*(_BYTE *)v36 & 4) != 0 )
      {
        v36 = *(_QWORD *)(v36 + 8);
        if ( v39 == v36 )
          goto LABEL_19;
      }
      else
      {
        while ( (*(_BYTE *)(v36 + 44) & 8) != 0 )
          v36 = *(_QWORD *)(v36 + 8);
        v36 = *(_QWORD *)(v36 + 8);
        if ( v39 == v36 )
        {
LABEL_19:
          v32 = v38;
LABEL_20:
          if ( v33-- == 0 )
          {
            v5 = v34;
            goto LABEL_22;
          }
          goto LABEL_9;
        }
      }
    }
    v95 = v38;
    v51 = sub_35994F0((__int64)v34, v36, v102, v33);
    sub_359F080(v34, v51, 0, v102, v33, a4);
    sub_2E31040((__int64 *)(v93 + 40), v51);
    v38 = v95;
    v52 = *(_QWORD *)(v95 + 48);
    *(_QWORD *)(v51 + 8) = v100;
    v52 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v51 = v52 | *(_QWORD *)v51 & 7LL;
    *(_QWORD *)(v52 + 8) = v51;
    v53 = v109;
    v54 = (__int64)v107;
    *(_QWORD *)(v95 + 48) = v51 | *(_QWORD *)(v95 + 48) & 7LL;
    if ( v53 )
    {
      v55 = (v53 - 1) & (((unsigned int)v51 >> 4) ^ ((unsigned int)v51 >> 9));
      v56 = (__int64 *)(v54 + 16LL * v55);
      v57 = *v56;
      if ( v51 == *v56 )
      {
LABEL_35:
        v58 = v56 + 1;
LABEL_36:
        *v58 = v36;
        goto LABEL_17;
      }
      v96 = 1;
      v61 = 0;
      while ( v57 != -4096 )
      {
        if ( !v61 && v57 == -8192 )
          v61 = v56;
        v55 = (v53 - 1) & (v96 + v55);
        v56 = (__int64 *)(v54 + 16LL * v55);
        v57 = *v56;
        if ( v51 == *v56 )
          goto LABEL_35;
        ++v96;
      }
      if ( !v61 )
        v61 = v56;
      ++v106;
      v62 = v108 + 1;
      if ( 4 * ((int)v108 + 1) < 3 * v53 )
      {
        if ( v53 - HIDWORD(v108) - v62 <= v53 >> 3 )
        {
          v91 = v38;
          sub_2E48800((__int64)&v106, v53);
          if ( !v109 )
            goto LABEL_112;
          v82 = 0;
          v38 = v91;
          v83 = (v109 - 1) & (((unsigned int)v51 >> 4) ^ ((unsigned int)v51 >> 9));
          v62 = v108 + 1;
          v84 = 1;
          v61 = &v107[2 * v83];
          v85 = *v61;
          if ( v51 != *v61 )
          {
            while ( v85 != -4096 )
            {
              if ( v85 == -8192 && !v82 )
                v82 = v61;
              v83 = (v109 - 1) & (v84 + v83);
              v61 = &v107[2 * v83];
              v85 = *v61;
              if ( v51 == *v61 )
                goto LABEL_47;
              ++v84;
            }
            if ( v82 )
              v61 = v82;
          }
        }
LABEL_47:
        LODWORD(v108) = v62;
        if ( *v61 != -4096 )
          --HIDWORD(v108);
        *v61 = v51;
        v58 = v61 + 1;
        *v58 = 0;
        goto LABEL_36;
      }
    }
    else
    {
      ++v106;
    }
    v88 = v38;
    v89 = v54;
    v97 = v53;
    v63 = (((((((2 * v53 - 1) | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 2)
            | (2 * v53 - 1)
            | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 4)
          | (((2 * v53 - 1) | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 2)
          | (2 * v53 - 1)
          | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 8)
        | (((((2 * v53 - 1) | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 2)
          | (2 * v53 - 1)
          | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 4)
        | (((2 * v53 - 1) | ((unsigned __int64)(2 * v53 - 1) >> 1)) >> 2)
        | (2 * v53 - 1)
        | ((unsigned __int64)(2 * v53 - 1) >> 1);
    v64 = ((v63 >> 16) | v63) + 1;
    if ( (unsigned int)v64 < 0x40 )
      LODWORD(v64) = 64;
    v109 = v64;
    v65 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v64, 8);
    v66 = v89;
    v107 = v65;
    v38 = v88;
    if ( v89 )
    {
      v108 = 0;
      v90 = 16LL * v97;
      v98 = (__int64 *)(v66 + v90);
      for ( i = &v65[2 * v109]; i != v65; v65 += 2 )
      {
        if ( v65 )
          *v65 = -4096;
      }
      v68 = (__int64 *)v66;
      if ( (__int64 *)v66 != v98 )
      {
        do
        {
          v69 = *v68;
          if ( *v68 != -8192 && v69 != -4096 )
          {
            if ( !v109 )
            {
              MEMORY[0] = *v68;
              BUG();
            }
            v70 = 1;
            v71 = 0;
            v72 = (v109 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
            v73 = &v107[2 * v72];
            v74 = *v73;
            if ( *v73 != v69 )
            {
              while ( v74 != -4096 )
              {
                if ( v74 == -8192 && !v71 )
                  v71 = v73;
                v72 = (v109 - 1) & (v70 + v72);
                v73 = &v107[2 * v72];
                v74 = *v73;
                if ( v69 == *v73 )
                  goto LABEL_63;
                ++v70;
              }
              if ( v71 )
                v73 = v71;
            }
LABEL_63:
            *v73 = v69;
            v73[1] = v68[1];
            LODWORD(v108) = v108 + 1;
          }
          v68 += 2;
        }
        while ( v98 != v68 );
        v38 = v88;
      }
      v99 = v38;
      sub_C7D6A0(v66, v90, 8);
      v75 = (__int64)v107;
      v76 = v109;
      v38 = v99;
      v62 = v108 + 1;
    }
    else
    {
      v108 = 0;
      v76 = v109;
      v75 = (__int64)&v65[2 * v109];
      if ( v65 != (_QWORD *)v75 )
      {
        v86 = v65;
        do
        {
          if ( v86 )
            *v86 = -4096;
          v86 += 2;
        }
        while ( (_QWORD *)v75 != v86 );
        v75 = (__int64)v65;
      }
      v62 = 1;
    }
    if ( !v76 )
    {
LABEL_112:
      LODWORD(v108) = v108 + 1;
      BUG();
    }
    v77 = v76 - 1;
    v78 = (v76 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v61 = (__int64 *)(v75 + 16LL * v78);
    v79 = *v61;
    if ( v51 != *v61 )
    {
      v80 = 1;
      v81 = 0;
      while ( v79 != -4096 )
      {
        if ( v79 == -8192 && !v81 )
          v81 = v61;
        v78 = v77 & (v80 + v78);
        v61 = (__int64 *)(v75 + 16LL * v78);
        v79 = *v61;
        if ( v51 == *v61 )
          goto LABEL_47;
        ++v80;
      }
      if ( v81 )
        v61 = v81;
    }
    goto LABEL_47;
  }
LABEL_24:
  sub_2E33690(v6, v5[6], a3);
  if ( (*(unsigned int (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v5[4] + 360LL))(v5[4], v5[7], 0) )
  {
    v46 = (__int64 *)v5[4];
    v104 = &v106;
    v105 = 0;
    v47 = *v46;
    v48 = *(_QWORD **)a5;
    v49 = v5[7];
    v103 = 0;
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD, __int64 *, _QWORD, __int64 *, _QWORD))(v47 + 368))(
      v46,
      v49,
      *v48,
      0,
      &v106,
      0,
      &v103,
      0);
    sub_9C6650(&v103);
    if ( v104 != &v106 )
      _libc_free((unsigned __int64)v104);
  }
  return sub_C7D6A0((__int64)v107, 16LL * v109, 8);
}
