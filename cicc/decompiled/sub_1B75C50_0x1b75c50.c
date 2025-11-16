// Function: sub_1B75C50
// Address: 0x1b75c50
//
__int64 __fastcall sub_1B75C50(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r15
  unsigned __int8 v11; // al
  _QWORD *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // edi
  __int64 v16; // rcx
  __int64 v17; // r9
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 **v23; // r13
  __int64 v24; // rax
  __int64 *v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 *v29; // rax
  int v30; // ecx
  int v31; // r10d
  __int64 v32; // r15
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r13d
  __int64 v37; // r12
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 *v42; // rcx
  int v43; // r9d
  __int64 *v44; // r14
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 v47; // r14
  _QWORD *v48; // rax
  __int64 v49; // r13
  unsigned int v50; // eax
  __int64 v51; // rdx
  __int64 v52; // r8
  _QWORD *v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 **v56; // r14
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned int v65; // r13d
  unsigned int v66; // eax
  __int64 v67; // r14
  unsigned int v68; // r12d
  __int64 v69; // rdx
  int v70; // r8d
  int v71; // r9d
  __int64 v72; // rax
  __int64 v73; // r13
  unsigned __int8 v74; // al
  __int64 v75; // rax
  __int64 v76; // r13
  _QWORD *v77; // rax
  __int64 (__fastcall *v78)(__int64, __int64); // r15
  __int64 v79; // rax
  _QWORD *v80; // r12
  __int64 v81; // rax
  _QWORD *v82; // rax
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // r14
  __int64 v85; // rax
  unsigned __int64 v86; // r8
  __int64 v87; // rdx
  _QWORD *v88; // rax
  _QWORD *v89; // r8
  _QWORD *v90; // rdx
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // r13
  unsigned __int64 v93; // rbx
  __int64 v94; // r12
  __int128 v95; // rdi
  __int64 v96; // [rsp+0h] [rbp-B0h]
  __int64 v97; // [rsp+8h] [rbp-A8h]
  __int64 **v98; // [rsp+10h] [rbp-A0h]
  __int64 v99; // [rsp+10h] [rbp-A0h]
  __int64 v100; // [rsp+10h] [rbp-A0h]
  __int64 v101; // [rsp+10h] [rbp-A0h]
  int v102; // [rsp+18h] [rbp-98h]
  __int64 v103; // [rsp+20h] [rbp-90h]
  __int64 v104; // [rsp+20h] [rbp-90h]
  unsigned int v105; // [rsp+28h] [rbp-88h]
  __int64 v106; // [rsp+28h] [rbp-88h]
  __int64 **v107; // [rsp+28h] [rbp-88h]
  __int64 *v108; // [rsp+30h] [rbp-80h] BYREF
  __int64 v109; // [rsp+38h] [rbp-78h]
  _WORD v110[56]; // [rsp+40h] [rbp-70h] BYREF

  v6 = a1;
  v7 = (_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 16));
  v8 = *(unsigned int *)(*v7 + 24LL);
  if ( (_DWORD)v8 )
  {
    v14 = *(_QWORD *)(*v7 + 8LL);
    v15 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = v14 + ((unsigned __int64)v15 << 6);
    v17 = *(_QWORD *)(v16 + 24);
    if ( v17 == a2 )
    {
LABEL_15:
      if ( v16 != v14 + (v8 << 6) )
        return *(_QWORD *)(v16 + 56);
    }
    else
    {
      v30 = 1;
      while ( v17 != -8 )
      {
        v31 = v30 + 1;
        v15 = (v8 - 1) & (v30 + v15);
        v16 = v14 + ((unsigned __int64)v15 << 6);
        v17 = *(_QWORD *)(v16 + 24);
        if ( v17 == a2 )
          goto LABEL_15;
        v30 = v31;
      }
    }
  }
  v9 = v7[1];
  if ( v9 )
  {
    v10 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 8LL))(v9, a2);
    if ( v10 )
    {
      v19 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
      v20 = v19[2];
      if ( v10 == v20 )
        return v10;
      if ( v20 != -8 && v20 != 0 && v20 != -16 )
        sub_1649B30(v19);
      v19[2] = v10;
      goto LABEL_23;
    }
  }
  v11 = *(_BYTE *)(a2 + 16);
  if ( v11 <= 3u )
  {
    if ( (*(_BYTE *)v6 & 8) == 0 )
      goto LABEL_6;
    return 0;
  }
  if ( v11 == 20 )
  {
    v21 = sub_15EAB70(a2);
    v22 = *(_QWORD *)(v6 + 8);
    if ( !v22
      || (v23 = (__int64 **)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v22 + 24LL))(v22, v21),
          v23 == (__int64 **)sub_15EAB70(a2)) )
    {
      v10 = a2;
    }
    else
    {
      v10 = sub_15EE570(
              v23,
              *(_BYTE **)(a2 + 24),
              *(_QWORD *)(a2 + 32),
              *(_BYTE **)(a2 + 56),
              *(_QWORD *)(a2 + 64),
              *(_BYTE *)(a2 + 96),
              *(_BYTE *)(a2 + 97),
              0);
    }
    v19 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), v10);
    v24 = v19[2];
    if ( v24 == v10 )
      return v10;
    if ( v24 != 0 && v24 != -8 )
    {
LABEL_32:
      if ( v24 != -16 )
        sub_1649B30(v19);
    }
LABEL_34:
    v19[2] = v10;
    if ( !v10 )
      return 0;
LABEL_23:
    if ( v10 != -8 )
    {
      if ( v10 != -16 )
        sub_164C220((__int64)v19);
      return v10;
    }
    return -8;
  }
  if ( v11 != 19 )
  {
    if ( v11 > 0x10u )
      return 0;
    if ( v11 == 4 )
    {
      v32 = sub_1B75C50(v6, *(_QWORD *)(a2 - 48));
      if ( v32 + 72 == (*(_QWORD *)(v32 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v106 = *(_QWORD *)(a2 - 24);
        v110[0] = 257;
        v47 = sub_16498A0(a2);
        v48 = (_QWORD *)sub_22077B0(64);
        v49 = (__int64)v48;
        if ( v48 )
          sub_157FB60(v48, v47, (__int64)&v108, 0, 0);
        v50 = *(_DWORD *)(v6 + 192);
        v51 = *(unsigned int *)(v6 + 196);
        if ( v50 >= (unsigned int)v51 )
        {
          v83 = (((((v51 + 2) | ((unsigned __int64)(v51 + 2) >> 1)) >> 2)
                | (v51 + 2)
                | ((unsigned __int64)(v51 + 2) >> 1)) >> 4)
              | (((v51 + 2) | ((unsigned __int64)(v51 + 2) >> 1)) >> 2)
              | (v51 + 2)
              | ((unsigned __int64)(v51 + 2) >> 1);
          v84 = ((v83 >> 8) | v83 | (((v83 >> 8) | v83) >> 16) | (((v83 >> 8) | v83) >> 32)) + 1;
          if ( v84 > 0xFFFFFFFF )
            v84 = 0xFFFFFFFFLL;
          v104 = malloc(16 * v84);
          if ( !v104 )
            sub_16BD1C0("Allocation failed", 1u);
          v85 = *(unsigned int *)(v6 + 192);
          v86 = *(_QWORD *)(v6 + 184);
          v87 = 16 * v85;
          if ( 16 * v85 )
          {
            v88 = (_QWORD *)v104;
            v89 = (_QWORD *)(v86 + 8);
            v90 = (_QWORD *)(v104 + v87);
            do
            {
              if ( v88 )
              {
                *v88 = *(v89 - 1);
                v88[1] = *v89;
                *v89 = 0;
              }
              v88 += 2;
              v89 += 2;
            }
            while ( v88 != v90 );
            v86 = *(_QWORD *)(v6 + 184);
            v85 = *(unsigned int *)(v6 + 192);
          }
          v91 = v86 + 16 * v85;
          if ( v86 != v91 )
          {
            v97 = a2;
            v102 = v84;
            v84 = v49;
            v92 = v86;
            v100 = v6;
            v93 = v91;
            do
            {
              v94 = *(_QWORD *)(v93 - 8);
              v93 -= 16LL;
              if ( v94 )
              {
                sub_157EF40(v94);
                j_j___libc_free_0(v94, 64);
              }
            }
            while ( v92 != v93 );
            v6 = v100;
            v49 = v84;
            a2 = v97;
            LODWORD(v84) = v102;
            v86 = *(_QWORD *)(v100 + 184);
          }
          if ( v86 != v6 + 200 )
            _libc_free(v86);
          *(_DWORD *)(v6 + 196) = v84;
          *(_QWORD *)(v6 + 184) = v104;
          v50 = *(_DWORD *)(v6 + 192);
        }
        v52 = *(_QWORD *)(v6 + 184);
        v53 = (_QWORD *)(v52 + 16LL * v50);
        if ( v53 )
        {
          v53[1] = v49;
          *v53 = v106;
          v52 = *(_QWORD *)(v6 + 184);
          v54 = (unsigned int)(*(_DWORD *)(v6 + 192) + 1);
          *(_DWORD *)(v6 + 192) = v54;
        }
        else
        {
          v54 = v50 + 1;
          *(_DWORD *)(v6 + 192) = v54;
          if ( v49 )
          {
            sub_157EF40(v49);
            j_j___libc_free_0(v49, 64);
            v54 = *(unsigned int *)(v6 + 192);
            v52 = *(_QWORD *)(v6 + 184);
          }
        }
        v33 = *(_QWORD *)(v52 + 16 * v54 - 8);
        if ( v33 )
          goto LABEL_52;
      }
      else
      {
        v33 = sub_1B75C50(v6, *(_QWORD *)(a2 - 24));
        if ( v33 )
        {
LABEL_52:
          v34 = sub_159BBF0(v32, v33);
          goto LABEL_53;
        }
      }
      v33 = *(_QWORD *)(a2 - 24);
      goto LABEL_52;
    }
    v105 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v105 )
    {
      v35 = a2;
      v36 = 0;
      v37 = 0;
      v38 = v35;
      while ( 1 )
      {
        v39 = (*(_BYTE *)(v38 + 23) & 0x40) != 0
            ? *(_QWORD *)(v38 - 8)
            : v38 - 24LL * (*(_DWORD *)(v38 + 20) & 0xFFFFFFF);
        v40 = *(_QWORD *)(v39 + v37);
        v41 = sub_1B75C50(v6, v40);
        if ( !v41 )
          return 0;
        if ( v40 != v41 )
        {
          v55 = *(_QWORD *)(v6 + 8);
          v56 = *(__int64 ***)v38;
          v57 = v41;
          a2 = v38;
          if ( !v55 )
            goto LABEL_83;
          goto LABEL_82;
        }
        ++v36;
        v37 += 24;
        if ( v36 == v105 )
        {
          v57 = v41;
          a2 = v38;
          goto LABEL_115;
        }
      }
    }
    v36 = 0;
    v57 = 0;
LABEL_115:
    v55 = *(_QWORD *)(v6 + 8);
    if ( !v55 )
      goto LABEL_130;
    v56 = *(__int64 ***)a2;
LABEL_82:
    v103 = v57;
    v58 = (*(__int64 (__fastcall **)(__int64, __int64 **))(*(_QWORD *)v55 + 24LL))(v55, v56);
    v57 = v103;
    v56 = (__int64 **)v58;
    if ( v105 == v36 && v58 == *(_QWORD *)a2 )
    {
LABEL_130:
      v82 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
      return sub_1B74FE0(v82, a2);
    }
LABEL_83:
    v108 = (__int64 *)v110;
    v109 = 0x800000000LL;
    if ( v105 > 8uLL )
    {
      v99 = v57;
      sub_16CD150((__int64)&v108, v110, v105, 8, v57, v43);
      v57 = v99;
    }
    if ( v36 )
    {
      v98 = v56;
      v59 = (unsigned int)v109;
      v60 = 0;
      v61 = v57;
      do
      {
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v62 = *(_QWORD *)(a2 - 8);
        else
          v62 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v63 = *(_QWORD *)(v62 + v60);
        if ( (unsigned int)v59 >= HIDWORD(v109) )
        {
          v96 = v63;
          sub_16CD150((__int64)&v108, v110, 0, 8, v57, v43);
          v59 = (unsigned int)v109;
          v63 = v96;
        }
        v42 = v108;
        v60 += 24;
        v108[v59] = v63;
        v59 = (unsigned int)(v109 + 1);
        LODWORD(v109) = v109 + 1;
      }
      while ( 24LL * v36 != v60 );
      v57 = v61;
      v56 = v98;
    }
    if ( v105 != v36 )
    {
      v64 = (unsigned int)v109;
      if ( (unsigned int)v109 >= HIDWORD(v109) )
      {
        v101 = v57;
        sub_16CD150((__int64)&v108, v110, 0, 8, v57, v43);
        v64 = (unsigned int)v109;
        v57 = v101;
      }
      v65 = v36 + 1;
      v108[v64] = v57;
      v66 = v105;
      LODWORD(v109) = v109 + 1;
      if ( v65 != v105 )
      {
        v107 = v56;
        v67 = a2;
        v68 = v66;
        while ( 1 )
        {
          v69 = (*(_BYTE *)(v67 + 23) & 0x40) != 0
              ? *(_QWORD *)(v67 - 8)
              : v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
          v10 = sub_1B75C50(v6, *(_QWORD *)(v69 + 24LL * v65));
          if ( !v10 )
            goto LABEL_112;
          v72 = (unsigned int)v109;
          if ( (unsigned int)v109 >= HIDWORD(v109) )
          {
            sub_16CD150((__int64)&v108, v110, 0, 8, v70, v71);
            v72 = (unsigned int)v109;
          }
          ++v65;
          v108[v72] = v10;
          LODWORD(v109) = v109 + 1;
          if ( v65 == v68 )
          {
            a2 = v67;
            v56 = v107;
            break;
          }
        }
      }
    }
    v73 = *(_QWORD *)(v6 + 8);
    if ( v73 )
    {
      v74 = *(_BYTE *)(a2 + 16);
      if ( v74 <= 0x17u )
      {
        if ( v74 != 5 )
          goto LABEL_134;
        if ( *(_WORD *)(a2 + 18) != 32 )
        {
          v73 = 0;
LABEL_119:
          v10 = sub_15A47B0(a2, (_BYTE **)v108, (unsigned int)v109, v56, 0, v73, a3, a4, a5);
          v80 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
          v81 = v80[2];
          if ( v10 != v81 )
          {
            if ( v81 != -8 && v81 != 0 && v81 != -16 )
              sub_1649B30(v80);
            v80[2] = v10;
            if ( v10 != -8 && v10 != 0 && v10 != -16 )
              sub_164C220((__int64)v80);
          }
          goto LABEL_112;
        }
      }
      else if ( v74 != 56 )
      {
        goto LABEL_110;
      }
      v78 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v73 + 24LL);
      v79 = sub_16348C0(a2);
      v73 = v78(v73, v79);
    }
    v74 = *(_BYTE *)(a2 + 16);
    if ( v74 == 5 )
      goto LABEL_119;
LABEL_134:
    switch ( v74 )
    {
      case 6u:
        *((_QWORD *)&v95 + 1) = v108;
        *(_QWORD *)&v95 = v56;
        v75 = sub_159DFD0(v95, (unsigned int)v109, (__int64)v42);
        goto LABEL_111;
      case 7u:
        v75 = sub_159F090(v56, v108, (unsigned int)v109, (__int64)v42);
        goto LABEL_111;
      case 8u:
        v75 = sub_15A01B0(v108, (unsigned int)v109);
        goto LABEL_111;
      case 9u:
        v75 = sub_1599EF0(v56);
        goto LABEL_111;
      case 0xAu:
        v75 = sub_1598F00(v56);
        goto LABEL_111;
    }
LABEL_110:
    v75 = sub_1599A20(v56);
LABEL_111:
    v76 = v75;
    v77 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
    v10 = sub_1B74FE0(v77, v76);
LABEL_112:
    if ( v108 != (__int64 *)v110 )
      _libc_free((unsigned __int64)v108);
    return v10;
  }
  v25 = *(__int64 **)(a2 + 24);
  if ( *(_BYTE *)v25 != 2 )
  {
    if ( (*(_BYTE *)v6 & 1) != 0 )
    {
      v12 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
      v13 = v12[2];
      if ( v13 != a2 )
      {
        if ( v13 == -8 || v13 == 0 )
          goto LABEL_10;
        goto LABEL_8;
      }
      return a2;
    }
    sub_1B76840(&v108, v6, *(_QWORD *)(a2 + 24));
    if ( (_BYTE)v109 )
      v44 = v108;
    else
      v44 = (__int64 *)sub_1B785E0(v6, v25);
    if ( v25 == v44 )
    {
LABEL_6:
      v12 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
      v13 = v12[2];
      if ( v13 != a2 )
      {
        if ( v13 == 0 || v13 == -8 )
          goto LABEL_10;
LABEL_8:
        if ( v13 != -16 )
          sub_1649B30(v12);
LABEL_10:
        v12[2] = a2;
        if ( a2 != -8 )
        {
          if ( a2 != -16 )
            sub_164C220((__int64)v12);
          return a2;
        }
        return -8;
      }
      return a2;
    }
    v45 = (__int64 *)sub_16498A0(a2);
    v34 = sub_1628DA0(v45, (__int64)v44);
LABEL_53:
    v10 = v34;
    v19 = sub_1B758F0(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * *(unsigned int *)(v6 + 16)), a2);
    v24 = v19[2];
    if ( v10 == v24 )
      return v10;
    if ( v24 != -8 && v24 != 0 )
      goto LABEL_32;
    goto LABEL_34;
  }
  v26 = sub_1B75C50(v6, v25[17]);
  if ( v26 )
  {
    if ( v25[17] == a2 )
      return a2;
    v27 = (__int64)sub_1624210(v26);
  }
  else
  {
    if ( (*(_BYTE *)v6 & 2) != 0 )
      return 0;
    v46 = (__int64 *)sub_16498A0(a2);
    v27 = sub_1627350(v46, 0, 0, 0, 1);
  }
  v28 = v27;
  v29 = (__int64 *)sub_16498A0(a2);
  return sub_1628DA0(v29, v28);
}
