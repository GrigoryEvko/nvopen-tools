// Function: sub_1BEA4B0
// Address: 0x1bea4b0
//
__int64 __fastcall sub_1BEA4B0(__int64 a1, __int64 a2)
{
  int v3; // r15d
  __int64 v4; // r12
  _QWORD *v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // r14
  __int64 v9; // r15
  __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  int v19; // r14d
  __int64 *v20; // r15
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // r8
  _BYTE *v24; // rsi
  int v25; // r8d
  __int64 *v26; // r12
  __int64 *v27; // r15
  int v28; // r9d
  __int64 *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rsi
  int v33; // r10d
  unsigned int v34; // edx
  __int64 v35; // rcx
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 *v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // r8
  unsigned int v41; // r14d
  unsigned int v42; // edi
  _QWORD *v43; // rbx
  __int64 v44; // rcx
  __int64 v45; // r12
  __int64 v46; // rdi
  _QWORD *v47; // rbx
  _QWORD *v48; // r12
  unsigned __int64 v49; // rdi
  __int64 result; // rax
  char *v51; // rdx
  unsigned int v52; // eax
  _QWORD *v53; // rbx
  _QWORD *v54; // r12
  unsigned __int64 v55; // rdi
  _QWORD *v56; // r12
  __int64 v57; // r14
  __int64 v58; // rdi
  int v59; // edx
  int v60; // r10d
  int v61; // r10d
  __int64 v62; // r11
  unsigned int v63; // ecx
  int v64; // edi
  _QWORD *v65; // rax
  __int64 v66; // r9
  int v67; // ebx
  unsigned int v68; // eax
  _QWORD *v69; // rdi
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rdi
  _QWORD *v72; // rax
  __int64 v73; // rdx
  _QWORD *i; // rdx
  int v75; // r11d
  int v76; // ebx
  int v77; // r8d
  int v78; // r8d
  __int64 v79; // r9
  _QWORD *v80; // r10
  unsigned int v81; // r14d
  int v82; // ecx
  __int64 v83; // rsi
  _QWORD *v84; // rax
  int v85; // r8d
  _QWORD *v86; // rsi
  __int64 v87; // [rsp+8h] [rbp-328h]
  __int64 *v89; // [rsp+30h] [rbp-300h]
  __int64 v90; // [rsp+38h] [rbp-2F8h]
  __int64 *v91; // [rsp+38h] [rbp-2F8h]
  __int64 *v92; // [rsp+38h] [rbp-2F8h]
  __int64 v93; // [rsp+50h] [rbp-2E0h] BYREF
  __int64 v94; // [rsp+58h] [rbp-2D8h] BYREF
  _QWORD *v95; // [rsp+60h] [rbp-2D0h] BYREF
  _BYTE *v96; // [rsp+68h] [rbp-2C8h]
  _BYTE *v97; // [rsp+70h] [rbp-2C0h]
  __int64 v98; // [rsp+78h] [rbp-2B8h] BYREF
  _QWORD *v99; // [rsp+80h] [rbp-2B0h]
  __int64 v100; // [rsp+88h] [rbp-2A8h]
  unsigned int v101; // [rsp+90h] [rbp-2A0h]
  __int64 v102; // [rsp+98h] [rbp-298h]
  __int64 *v103; // [rsp+A0h] [rbp-290h] BYREF
  int v104; // [rsp+A8h] [rbp-288h]
  char v105; // [rsp+B0h] [rbp-280h] BYREF
  char *v106; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v107; // [rsp+F8h] [rbp-238h]
  _QWORD v108[70]; // [rsp+100h] [rbp-230h] BYREF

  v3 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 64);
  if ( !v3 && !*(_DWORD *)(a1 + 44) )
    goto LABEL_17;
  v5 = *(_QWORD **)(a1 + 32);
  v6 = 4 * v3;
  v7 = *(unsigned int *)(a1 + 48);
  v8 = &v5[2 * v7];
  if ( (unsigned int)(4 * v3) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v7 <= v6 )
  {
    while ( v5 != v8 )
    {
      if ( *v5 != -8 )
      {
        if ( *v5 != -16 )
        {
          v9 = v5[1];
          if ( v9 )
          {
            v10 = *(_QWORD *)(v9 + 24);
            if ( v10 )
              j_j___libc_free_0(v10, *(_QWORD *)(v9 + 40) - v10);
            j_j___libc_free_0(v9, 56);
          }
        }
        *v5 = -8;
      }
      v5 += 2;
    }
  }
  else
  {
    v90 = *(_QWORD *)(a1 + 64);
    v56 = &v5[2 * v7];
    do
    {
      if ( *v5 != -16 && *v5 != -8 )
      {
        v57 = v5[1];
        if ( v57 )
        {
          v58 = *(_QWORD *)(v57 + 24);
          if ( v58 )
            j_j___libc_free_0(v58, *(_QWORD *)(v57 + 40) - v58);
          j_j___libc_free_0(v57, 56);
        }
      }
      v5 += 2;
    }
    while ( v5 != v56 );
    v4 = v90;
    v59 = *(_DWORD *)(a1 + 48);
    if ( v3 )
    {
      v67 = 64;
      if ( v3 != 1 )
      {
        _BitScanReverse(&v68, v3 - 1);
        v67 = 1 << (33 - (v68 ^ 0x1F));
        if ( v67 < 64 )
          v67 = 64;
      }
      v69 = *(_QWORD **)(a1 + 32);
      if ( v67 == v59 )
      {
        *(_QWORD *)(a1 + 40) = 0;
        v84 = &v69[2 * (unsigned int)v67];
        do
        {
          if ( v69 )
            *v69 = -8;
          v69 += 2;
        }
        while ( v84 != v69 );
      }
      else
      {
        j___libc_free_0(v69);
        v70 = ((((((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
                 | (4 * v67 / 3u + 1)
                 | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
               | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
               | (4 * v67 / 3u + 1)
               | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
               | (4 * v67 / 3u + 1)
               | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
             | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
             | (4 * v67 / 3u + 1)
             | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 16;
        v71 = (v70
             | (((((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
                 | (4 * v67 / 3u + 1)
                 | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
               | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
               | (4 * v67 / 3u + 1)
               | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
               | (4 * v67 / 3u + 1)
               | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 4)
             | (((4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1)) >> 2)
             | (4 * v67 / 3u + 1)
             | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 48) = v71;
        v72 = (_QWORD *)sub_22077B0(16 * v71);
        v73 = *(unsigned int *)(a1 + 48);
        *(_QWORD *)(a1 + 40) = 0;
        *(_QWORD *)(a1 + 32) = v72;
        for ( i = &v72[2 * v73]; i != v72; v72 += 2 )
        {
          if ( v72 )
            *v72 = -8;
        }
      }
      goto LABEL_17;
    }
    if ( v59 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_DWORD *)(a1 + 48) = 0;
      goto LABEL_17;
    }
  }
  *(_QWORD *)(a1 + 40) = 0;
LABEL_17:
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 76) = 0;
  *(_QWORD *)(a1 + 64) = v4;
  v96 = 0;
  v97 = 0;
  v11 = (_QWORD *)sub_22077B0(8);
  v106 = (char *)v108;
  v12 = (__int64)(v11 + 1);
  v95 = v11;
  *v11 = 0;
  v13 = *(_QWORD *)(a1 + 64);
  v97 = (_BYTE *)v12;
  v14 = *(_QWORD *)(v13 + 112);
  v96 = (_BYTE *)v12;
  v98 = 0;
  v108[0] = v14;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v107 = 0x100000001LL;
  sub_1BE2200(a1, &v106, v12, v15, v16, v17);
  if ( v106 != (char *)v108 )
    _libc_free((unsigned __int64)v106);
  v18 = **(_QWORD **)a1;
  v106 = (char *)v108;
  v107 = 0x4000000001LL;
  v108[0] = v18;
  v51 = (char *)v108;
  v52 = 1;
  v87 = a1;
  v19 = 0;
  v20 = &v93;
  do
  {
    v21 = *(_QWORD *)&v51[8 * v52 - 8];
    LODWORD(v107) = v52 - 1;
    v93 = v21;
    v22 = sub_1BE8E40((__int64)&v98, v20);
    if ( !*((_DWORD *)v22 + 2) )
    {
      v23 = v93;
      *((_DWORD *)v22 + 2) = ++v19;
      v24 = v96;
      *((_DWORD *)v22 + 4) = v19;
      v22[3] = v23;
      if ( v24 == v97 )
      {
        sub_1BE3A50((__int64)&v95, v24, v20);
        v23 = v93;
      }
      else
      {
        if ( v24 )
        {
          *(_QWORD *)v24 = v23;
          v24 = v96;
          v23 = v93;
        }
        v96 = v24 + 8;
      }
      sub_1BE4A00((__int64)&v103, v23, v102);
      v25 = (int)v103;
      v26 = &v103[v104];
      if ( v103 != v26 )
      {
        v89 = v20;
        v27 = v103;
        while ( 1 )
        {
          v32 = *v27;
          v94 = *v27;
          if ( !v101 )
            goto LABEL_28;
          v28 = v101 - 1;
          v33 = 1;
          v34 = (v101 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v29 = &v99[9 * v34];
          v35 = *v29;
          if ( v32 != *v29 )
            break;
LABEL_35:
          if ( v29 == &v99[9 * v101] || !*((_DWORD *)v29 + 2) )
            goto LABEL_28;
          if ( v32 == v93 )
            goto LABEL_32;
          v31 = *((unsigned int *)v29 + 12);
          if ( (unsigned int)v31 >= *((_DWORD *)v29 + 13) )
            goto LABEL_39;
LABEL_31:
          *(_QWORD *)(v29[5] + 8 * v31) = v93;
          ++*((_DWORD *)v29 + 12);
LABEL_32:
          if ( v26 == ++v27 )
          {
            v20 = v89;
            v26 = v103;
            goto LABEL_45;
          }
        }
        while ( v35 != -8 )
        {
          v25 = v33 + 1;
          v34 = v28 & (v33 + v34);
          v29 = &v99[9 * v34];
          v35 = *v29;
          if ( v32 == *v29 )
            goto LABEL_35;
          ++v33;
        }
LABEL_28:
        v29 = sub_1BE8E40((__int64)&v98, &v94);
        v30 = (unsigned int)v107;
        if ( (unsigned int)v107 >= HIDWORD(v107) )
        {
          sub_16CD150((__int64)&v106, v108, 0, 8, v25, v28);
          v30 = (unsigned int)v107;
        }
        *(_QWORD *)&v106[8 * v30] = v94;
        LODWORD(v107) = v107 + 1;
        v31 = *((unsigned int *)v29 + 12);
        *((_DWORD *)v29 + 3) = v19;
        if ( (unsigned int)v31 < *((_DWORD *)v29 + 13) )
          goto LABEL_31;
LABEL_39:
        sub_16CD150((__int64)(v29 + 5), v29 + 7, 0, 8, v25, v28);
        v31 = *((unsigned int *)v29 + 12);
        goto LABEL_31;
      }
LABEL_45:
      if ( v26 != (__int64 *)&v105 )
        _libc_free((unsigned __int64)v26);
    }
    v52 = v107;
    v51 = v106;
  }
  while ( (_DWORD)v107 );
  if ( v106 != (char *)v108 )
    _libc_free((unsigned __int64)v106);
  sub_1BE9A60((__int64 *)&v95, v87, 0);
  if ( a2 )
    *(_BYTE *)(a2 + 144) = 1;
  if ( *(_DWORD *)(v87 + 8) )
  {
    v36 = **(_QWORD **)v87;
    v37 = sub_22077B0(56);
    v38 = (__int64 *)v37;
    if ( v37 )
    {
      *(_QWORD *)v37 = v36;
      *(_QWORD *)(v37 + 8) = 0;
      *(_DWORD *)(v37 + 16) = 0;
      *(_QWORD *)(v37 + 24) = 0;
      *(_QWORD *)(v37 + 32) = 0;
      *(_QWORD *)(v37 + 40) = 0;
      *(_QWORD *)(v37 + 48) = -1;
    }
    v39 = *(_DWORD *)(v87 + 48);
    if ( v39 )
    {
      v40 = *(_QWORD *)(v87 + 32);
      v41 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
      v42 = (v39 - 1) & v41;
      v43 = (_QWORD *)(v40 + 16LL * v42);
      v44 = *v43;
      if ( v36 == *v43 )
      {
LABEL_56:
        v45 = v43[1];
        v43[1] = v38;
        if ( v45 )
        {
          v46 = *(_QWORD *)(v45 + 24);
          if ( v46 )
            j_j___libc_free_0(v46, *(_QWORD *)(v45 + 40) - v46);
          j_j___libc_free_0(v45, 56);
          v38 = (__int64 *)v43[1];
        }
LABEL_60:
        *(_QWORD *)(v87 + 56) = v38;
        sub_1BE9420((__int64)&v95, v87, v38);
        if ( v101 )
        {
          v47 = v99;
          v48 = &v99[9 * v101];
          do
          {
            if ( *v47 != -16 && *v47 != -8 )
            {
              v49 = v47[5];
              if ( (_QWORD *)v49 != v47 + 7 )
                _libc_free(v49);
            }
            v47 += 9;
          }
          while ( v48 != v47 );
        }
        goto LABEL_67;
      }
      v75 = 1;
      v65 = 0;
      while ( v44 != -8 )
      {
        if ( !v65 && v44 == -16 )
          v65 = v43;
        v42 = (v39 - 1) & (v75 + v42);
        v43 = (_QWORD *)(v40 + 16LL * v42);
        v44 = *v43;
        if ( v36 == *v43 )
          goto LABEL_56;
        ++v75;
      }
      if ( !v65 )
        v65 = v43;
      v76 = *(_DWORD *)(v87 + 40);
      ++*(_QWORD *)(v87 + 24);
      v64 = v76 + 1;
      if ( 4 * (v76 + 1) < 3 * v39 )
      {
        if ( v39 - *(_DWORD *)(v87 + 44) - v64 > v39 >> 3 )
        {
LABEL_92:
          *(_DWORD *)(v87 + 40) = v64;
          if ( *v65 != -8 )
            --*(_DWORD *)(v87 + 44);
          *v65 = v36;
          v65[1] = v38;
          goto LABEL_60;
        }
        v92 = v38;
        sub_1BE8590(v87 + 24, v39);
        v77 = *(_DWORD *)(v87 + 48);
        if ( v77 )
        {
          v78 = v77 - 1;
          v79 = *(_QWORD *)(v87 + 32);
          v80 = 0;
          v81 = v78 & v41;
          v38 = v92;
          v82 = 1;
          v64 = *(_DWORD *)(v87 + 40) + 1;
          v65 = (_QWORD *)(v79 + 16LL * v81);
          v83 = *v65;
          if ( v36 != *v65 )
          {
            while ( v83 != -8 )
            {
              if ( !v80 && v83 == -16 )
                v80 = v65;
              v81 = v78 & (v82 + v81);
              v65 = (_QWORD *)(v79 + 16LL * v81);
              v83 = *v65;
              if ( v36 == *v65 )
                goto LABEL_92;
              ++v82;
            }
            if ( v80 )
              v65 = v80;
          }
          goto LABEL_92;
        }
LABEL_141:
        ++*(_DWORD *)(v87 + 40);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v87 + 24);
    }
    v91 = v38;
    sub_1BE8590(v87 + 24, 2 * v39);
    v60 = *(_DWORD *)(v87 + 48);
    if ( v60 )
    {
      v61 = v60 - 1;
      v62 = *(_QWORD *)(v87 + 32);
      v38 = v91;
      v63 = v61 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v64 = *(_DWORD *)(v87 + 40) + 1;
      v65 = (_QWORD *)(v62 + 16LL * v63);
      v66 = *v65;
      if ( v36 != *v65 )
      {
        v85 = 1;
        v86 = 0;
        while ( v66 != -8 )
        {
          if ( v66 == -16 && !v86 )
            v86 = v65;
          v63 = v61 & (v85 + v63);
          v65 = (_QWORD *)(v62 + 16LL * v63);
          v66 = *v65;
          if ( v36 == *v65 )
            goto LABEL_92;
          ++v85;
        }
        if ( v86 )
          v65 = v86;
      }
      goto LABEL_92;
    }
    goto LABEL_141;
  }
  if ( v101 )
  {
    v53 = v99;
    v54 = &v99[9 * v101];
    do
    {
      if ( *v53 != -8 && *v53 != -16 )
      {
        v55 = v53[5];
        if ( (_QWORD *)v55 != v53 + 7 )
          _libc_free(v55);
      }
      v53 += 9;
    }
    while ( v54 != v53 );
  }
LABEL_67:
  result = j___libc_free_0(v99);
  if ( v95 )
    return j_j___libc_free_0(v95, v97 - (_BYTE *)v95);
  return result;
}
