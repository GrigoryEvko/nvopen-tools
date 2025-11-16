// Function: sub_1B7E8C0
// Address: 0x1b7e8c0
//
__int64 **__fastcall sub_1B7E8C0(__int64 a1, __int64 a2, __int64 a3, __int64 **a4, __int64 a5)
{
  __int64 v5; // r14
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rbx
  __int64 *v12; // r12
  _QWORD *v13; // rax
  unsigned int v14; // r8d
  _QWORD *v15; // r15
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // rax
  int v19; // r8d
  __int64 *v20; // r11
  __int64 *v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // rdi
  unsigned __int64 *v26; // rbx
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  char v32; // al
  char *v33; // rdx
  bool v34; // zf
  int v35; // eax
  __int64 v36; // rax
  int v37; // r9d
  __int64 v38; // rdx
  int v39; // r15d
  __int64 v40; // r12
  __int64 v41; // rax
  int v42; // r8d
  __int64 v43; // rax
  __int64 v44; // r8
  int v45; // r9d
  __int64 v46; // rax
  int v47; // eax
  __int64 v49; // rax
  int v50; // r9d
  unsigned int v51; // edx
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 v54; // r14
  int v55; // r8d
  __int64 v56; // rax
  __int64 v57; // r8
  int v58; // r9d
  __int64 v59; // rax
  __int64 v60; // rdx
  int v61; // eax
  char *v62; // rdx
  __int64 v63; // rbx
  __int64 *v64; // r12
  __int64 v65; // rdi
  unsigned __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rsi
  __int64 v70; // rdx
  unsigned __int8 *v71; // rsi
  char v72; // al
  _DWORD *v73; // r11
  __int64 v74; // r10
  __int64 *v75; // r14
  __int64 *v76; // rbx
  __int64 *v77; // rax
  __int64 v78; // r15
  __int64 v79; // rdi
  unsigned __int64 *v80; // r12
  __int64 v81; // rax
  unsigned __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  __int64 v86; // rax
  __int64 *v87; // rax
  __int64 v88; // [rsp+8h] [rbp-C8h]
  int v89; // [rsp+10h] [rbp-C0h]
  __int64 v90; // [rsp+10h] [rbp-C0h]
  int v91; // [rsp+18h] [rbp-B8h]
  __int64 v92; // [rsp+20h] [rbp-B0h]
  unsigned int v93; // [rsp+28h] [rbp-A8h]
  __int64 v94; // [rsp+28h] [rbp-A8h]
  unsigned int v95; // [rsp+28h] [rbp-A8h]
  __int64 v96; // [rsp+30h] [rbp-A0h]
  __int64 v97; // [rsp+30h] [rbp-A0h]
  unsigned int v98; // [rsp+30h] [rbp-A0h]
  unsigned __int64 *v99; // [rsp+30h] [rbp-A0h]
  __int64 v100; // [rsp+30h] [rbp-A0h]
  __int64 v101; // [rsp+38h] [rbp-98h]
  int v102; // [rsp+38h] [rbp-98h]
  __int64 v103; // [rsp+38h] [rbp-98h]
  __int64 v105; // [rsp+48h] [rbp-88h]
  const void *v106; // [rsp+48h] [rbp-88h]
  unsigned __int8 *v107; // [rsp+58h] [rbp-78h] BYREF
  __int64 v108[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v109; // [rsp+70h] [rbp-60h]
  char *v110; // [rsp+80h] [rbp-50h] BYREF
  char *v111; // [rsp+88h] [rbp-48h]
  __int16 v112; // [rsp+90h] [rbp-40h]

  v5 = a1;
  v8 = *(unsigned __int8 *)(a3 + 8);
  v9 = 100990;
  v105 = a5;
  if ( _bittest64(&v9, v8) )
  {
    v10 = *(_BYTE *)(a5 + 16);
    v109 = 257;
    if ( v10 )
    {
      if ( v10 == 1 )
      {
        v110 = ".aggrsplit";
        v112 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 17) == 1 )
        {
          v62 = *(char **)a5;
        }
        else
        {
          v62 = (char *)a5;
          v10 = 2;
        }
        v110 = v62;
        v111 = ".aggrsplit";
        LOBYTE(v112) = v10;
        HIBYTE(v112) = 3;
      }
    }
    else
    {
      v112 = 256;
    }
    v11 = *(_QWORD *)(a1 + 96);
    v12 = *(__int64 **)(a1 + 40);
    v101 = *(unsigned int *)(a1 + 48);
    v96 = *(_QWORD *)(a1 + 88);
    if ( !v11 )
    {
      v86 = **(_QWORD **)(a1 + 88);
      if ( *(_BYTE *)(v86 + 8) == 16 )
        v86 = **(_QWORD **)(v86 + 16);
      v11 = *(_QWORD *)(v86 + 24);
    }
    v93 = *(_DWORD *)(a1 + 48) + 1;
    v13 = sub_1648A60(72, v93);
    v14 = v93;
    v15 = v13;
    if ( v13 )
    {
      v92 = (__int64)v13;
      v94 = (__int64)&v13[-3 * v93];
      v16 = *(_QWORD *)v96;
      if ( *(_BYTE *)(*(_QWORD *)v96 + 8LL) == 16 )
        v16 = **(_QWORD **)(v16 + 16);
      v89 = v14;
      v91 = *(_DWORD *)(v16 + 8) >> 8;
      v17 = (__int64 *)sub_15F9F50(v11, (__int64)v12, v101);
      v18 = (__int64 *)sub_1646BA0(v17, v91);
      v19 = v89;
      v20 = v18;
      if ( *(_BYTE *)(*(_QWORD *)v96 + 8LL) == 16 )
      {
        v87 = sub_16463B0(v18, *(_QWORD *)(*(_QWORD *)v96 + 32LL));
        v19 = v89;
        v20 = v87;
      }
      else
      {
        v21 = &v12[v101];
        if ( v12 != v21 )
        {
          v22 = v12;
          while ( 1 )
          {
            v23 = *(_QWORD *)*v22;
            if ( *(_BYTE *)(v23 + 8) == 16 )
              break;
            if ( v21 == ++v22 )
              goto LABEL_14;
          }
          v24 = sub_16463B0(v20, *(_QWORD *)(v23 + 32));
          v19 = v89;
          v20 = v24;
        }
      }
LABEL_14:
      sub_15F1EA0((__int64)v15, (__int64)v20, 32, v94, v19, 0);
      v15[7] = v11;
      v15[8] = sub_15F9F50(v11, (__int64)v12, v101);
      sub_15F9CE0((__int64)v15, v96, v12, v101, (__int64)&v110);
    }
    else
    {
      v92 = 0;
    }
    sub_15FA2E0((__int64)v15, 1);
    v25 = *(_QWORD *)(a2 + 8);
    if ( v25 )
    {
      v26 = *(unsigned __int64 **)(a2 + 16);
      sub_157E9D0(v25 + 40, (__int64)v15);
      v27 = v15[3];
      v28 = *v26;
      v15[4] = v26;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      v15[3] = v28 | v27 & 7;
      *(_QWORD *)(v28 + 8) = v15 + 3;
      *v26 = *v26 & 7 | (unsigned __int64)(v15 + 3);
    }
    sub_164B780(v92, v108);
    v29 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 )
    {
      v107 = *(unsigned __int8 **)a2;
      sub_1623A60((__int64)&v107, v29, 2);
      v30 = v15[6];
      if ( v30 )
        sub_161E7C0((__int64)(v15 + 6), v30);
      v31 = v107;
      v15[6] = v107;
      if ( v31 )
        sub_1623210((__int64)&v107, v31, (__int64)(v15 + 6));
    }
    v32 = *(_BYTE *)(v105 + 16);
    if ( v32 )
    {
      if ( v32 == 1 )
      {
        v110 = ".load";
        v112 = 259;
      }
      else
      {
        if ( *(_BYTE *)(v105 + 17) == 1 )
        {
          v33 = *(char **)v105;
        }
        else
        {
          v33 = (char *)v105;
          v32 = 2;
        }
        v110 = v33;
        v111 = ".load";
        LOBYTE(v112) = v32;
        HIBYTE(v112) = 3;
      }
    }
    else
    {
      v112 = 256;
    }
    v63 = -(__int64)(unsigned int)(*(_DWORD *)(v5 + 104) | *(_DWORD *)(v5 + 108))
        & (unsigned int)(*(_DWORD *)(v5 + 104) | *(_DWORD *)(v5 + 108));
    v64 = sub_1648A60(64, 1u);
    if ( v64 )
      sub_15F9210((__int64)v64, *(_QWORD *)(*v15 + 24LL), (__int64)v15, 0, 0, 0);
    v65 = *(_QWORD *)(a2 + 8);
    if ( v65 )
    {
      v99 = *(unsigned __int64 **)(a2 + 16);
      sub_157E9D0(v65 + 40, (__int64)v64);
      v66 = *v99;
      v67 = v64[3] & 7;
      v64[4] = (__int64)v99;
      v66 &= 0xFFFFFFFFFFFFFFF8LL;
      v64[3] = v66 | v67;
      *(_QWORD *)(v66 + 8) = v64 + 3;
      *v99 = *v99 & 7 | (unsigned __int64)(v64 + 3);
    }
    sub_164B780((__int64)v64, (__int64 *)&v110);
    v68 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 )
    {
      v108[0] = *(_QWORD *)a2;
      sub_1623A60((__int64)v108, v68, 2);
      v69 = v64[6];
      v70 = (__int64)(v64 + 6);
      if ( v69 )
      {
        sub_161E7C0((__int64)(v64 + 6), v69);
        v70 = (__int64)(v64 + 6);
      }
      v71 = (unsigned __int8 *)v108[0];
      v64[6] = v108[0];
      if ( v71 )
        sub_1623210((__int64)v108, v71, v70);
    }
    sub_15F8F50((__int64)v64, v63);
    v72 = *(_BYTE *)(v105 + 16);
    if ( v72 )
    {
      if ( v72 == 1 )
      {
        v108[0] = (__int64)".aggrsplitinsert";
        v109 = 259;
      }
      else
      {
        if ( *(_BYTE *)(v105 + 17) == 1 )
          v105 = *(_QWORD *)v105;
        else
          v72 = 2;
        LOBYTE(v109) = v72;
        HIBYTE(v109) = 3;
        v108[0] = v105;
        v108[1] = (__int64)".aggrsplitinsert";
      }
    }
    else
    {
      v109 = 256;
    }
    v73 = *(_DWORD **)(v5 + 8);
    v74 = *(unsigned int *)(v5 + 16);
    v75 = *a4;
    if ( *((_BYTE *)*a4 + 16) > 0x10u || *((_BYTE *)v64 + 16) > 0x10u )
    {
      v100 = v74;
      v106 = v73;
      v112 = 257;
      v77 = sub_1648A60(88, 2u);
      v76 = v77;
      if ( v77 )
      {
        v78 = (__int64)v77;
        sub_15F1EA0((__int64)v77, *v75, 63, (__int64)(v77 - 6), 2, 0);
        v76[7] = (__int64)(v76 + 9);
        v76[8] = 0x400000000LL;
        sub_15FAD90((__int64)v76, (__int64)v75, (__int64)v64, v106, v100, (__int64)&v110);
      }
      else
      {
        v78 = 0;
      }
      v79 = *(_QWORD *)(a2 + 8);
      if ( v79 )
      {
        v80 = *(unsigned __int64 **)(a2 + 16);
        sub_157E9D0(v79 + 40, (__int64)v76);
        v81 = v76[3];
        v82 = *v80;
        v76[4] = (__int64)v80;
        v82 &= 0xFFFFFFFFFFFFFFF8LL;
        v76[3] = v82 | v81 & 7;
        *(_QWORD *)(v82 + 8) = v76 + 3;
        *v80 = *v80 & 7 | (unsigned __int64)(v76 + 3);
      }
      sub_164B780(v78, v108);
      v83 = *(_QWORD *)a2;
      if ( *(_QWORD *)a2 )
      {
        v107 = *(unsigned __int8 **)a2;
        sub_1623A60((__int64)&v107, v83, 2);
        v84 = v76[6];
        if ( v84 )
          sub_161E7C0((__int64)(v76 + 6), v84);
        v85 = v107;
        v76[6] = (__int64)v107;
        if ( v85 )
          sub_1623210((__int64)&v107, v85, (__int64)(v76 + 6));
      }
    }
    else
    {
      v76 = (__int64 *)sub_15A3A20(*a4, v64, v73, v74, 0);
    }
    *a4 = v76;
    return a4;
  }
  else
  {
    v34 = (_BYTE)v8 == 14;
    v35 = *(_DWORD *)(a1 + 104);
    if ( v34 )
    {
      v95 = *(_DWORD *)(a1 + 104);
      *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v35) & -(*(_DWORD *)(a1 + 108) | v35);
      v36 = sub_127FA20(*(_QWORD *)a1, *(_QWORD *)(a3 + 24));
      v38 = *(_QWORD *)(a3 + 32);
      v102 = (unsigned __int64)(v36 + 7) >> 3;
      if ( (_DWORD)v38 )
      {
        v39 = 0;
        v40 = 0;
        v97 = (unsigned int)v38;
        v41 = *(unsigned int *)(a1 + 16);
        do
        {
          v42 = v40;
          if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v41 )
          {
            sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 4, v40, v37);
            v41 = *(unsigned int *)(a1 + 16);
            v42 = v40;
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v41) = v42;
          ++*(_DWORD *)(a1 + 16);
          v43 = sub_1643350(*(_QWORD **)(a2 + 24));
          v44 = sub_159C470(v43, v40, 0);
          v46 = *(unsigned int *)(a1 + 48);
          if ( (unsigned int)v46 >= *(_DWORD *)(a1 + 52) )
          {
            v88 = v44;
            sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 8, v44, v45);
            v46 = *(unsigned int *)(a1 + 48);
            v44 = v88;
          }
          ++v40;
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v46) = v44;
          ++*(_DWORD *)(a1 + 48);
          *(_DWORD *)(a1 + 108) = v39;
          sub_1B7E8C0(a1, a2, *(_QWORD *)(a3 + 24), a4, v105);
          v47 = *(_DWORD *)(a1 + 16);
          --*(_DWORD *)(a1 + 48);
          v39 += v102;
          v41 = (unsigned int)(v47 - 1);
          *(_DWORD *)(a1 + 16) = v41;
        }
        while ( v97 != v40 );
      }
      *(_DWORD *)(a1 + 104) = v95;
      return (__int64 **)v95;
    }
    else
    {
      v98 = *(_DWORD *)(a1 + 104);
      *(_DWORD *)(a1 + 104) = (*(_DWORD *)(a1 + 108) | v35) & -(*(_DWORD *)(a1 + 108) | v35);
      v49 = sub_15A9930(*(_QWORD *)a1, a3);
      v51 = *(_DWORD *)(a3 + 12);
      v52 = v49;
      if ( v51 )
      {
        v103 = v51;
        v53 = *(unsigned int *)(a1 + 16);
        v54 = 0;
        do
        {
          v55 = v54;
          if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v53 )
          {
            sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 4, v54, v50);
            v53 = *(unsigned int *)(a1 + 16);
            v55 = v54;
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v53) = v55;
          ++*(_DWORD *)(a1 + 16);
          v56 = sub_1643350(*(_QWORD **)(a2 + 24));
          v57 = sub_159C470(v56, v54, 0);
          v59 = *(unsigned int *)(a1 + 48);
          if ( (unsigned int)v59 >= *(_DWORD *)(a1 + 52) )
          {
            v90 = v57;
            sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 8, v57, v58);
            v59 = *(unsigned int *)(a1 + 48);
            v57 = v90;
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v59) = v57;
          ++*(_DWORD *)(a1 + 48);
          *(_DWORD *)(a1 + 108) = *(_QWORD *)(v52 + 8 * v54 + 16);
          v60 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v54++);
          sub_1B7E8C0(a1, a2, v60, a4, v105);
          v61 = *(_DWORD *)(a1 + 16);
          --*(_DWORD *)(a1 + 48);
          v53 = (unsigned int)(v61 - 1);
          *(_DWORD *)(a1 + 16) = v53;
        }
        while ( v103 != v54 );
        v5 = a1;
      }
      *(_DWORD *)(v5 + 104) = v98;
      return (__int64 **)v98;
    }
  }
}
