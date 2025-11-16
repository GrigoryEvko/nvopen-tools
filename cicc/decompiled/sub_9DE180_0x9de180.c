// Function: sub_9DE180
// Address: 0x9de180
//
__int64 *__fastcall sub_9DE180(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  char v8; // dl
  unsigned __int64 v9; // rax
  int v10; // r14d
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  int v13; // r15d
  unsigned int v14; // eax
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 *v18; // rsi
  int v19; // eax
  __int64 v20; // rcx
  char v21; // dl
  int v22; // edx
  char v23; // al
  char v25; // al
  bool v26; // dl
  __int64 *v27; // rsi
  int v28; // edx
  __int64 v29; // rcx
  char v30; // al
  int v31; // edx
  __int64 v32; // rcx
  unsigned int v33; // esi
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // r11
  __int64 v36; // r9
  unsigned int v37; // edi
  __int64 *v38; // rax
  __int64 v39; // r8
  unsigned __int64 v40; // rax
  char v41; // al
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rcx
  __int64 v45; // rdi
  bool v46; // zf
  unsigned __int64 v47; // rax
  _QWORD *v48; // rax
  unsigned __int64 v49; // rax
  int v50; // edi
  int v51; // edi
  int v52; // edi
  int v53; // edi
  int v54; // r9d
  __int64 *v55; // r8
  unsigned int i; // esi
  __int64 v57; // r10
  int v58; // edi
  int v59; // edi
  int v60; // r9d
  unsigned int v61; // esi
  __int64 *v62; // r8
  __int64 v63; // r10
  unsigned int v64; // esi
  unsigned int v65; // esi
  unsigned __int64 v66; // [rsp+8h] [rbp-3D8h]
  unsigned __int64 v67; // [rsp+10h] [rbp-3D0h]
  unsigned int v68; // [rsp+20h] [rbp-3C0h]
  int v69; // [rsp+20h] [rbp-3C0h]
  unsigned __int64 v70; // [rsp+20h] [rbp-3C0h]
  unsigned __int64 v71; // [rsp+20h] [rbp-3C0h]
  __int64 *v72; // [rsp+28h] [rbp-3B8h]
  unsigned __int64 v73; // [rsp+28h] [rbp-3B8h]
  __int64 v74; // [rsp+28h] [rbp-3B8h]
  unsigned int v75; // [rsp+30h] [rbp-3B0h]
  __int64 v76; // [rsp+30h] [rbp-3B0h]
  unsigned __int64 v77; // [rsp+38h] [rbp-3A8h]
  __int64 v78; // [rsp+40h] [rbp-3A0h]
  __int64 v79; // [rsp+40h] [rbp-3A0h]
  unsigned int v80; // [rsp+48h] [rbp-398h]
  unsigned __int64 v81; // [rsp+68h] [rbp-378h] BYREF
  unsigned __int64 v82; // [rsp+70h] [rbp-370h] BYREF
  char v83; // [rsp+78h] [rbp-368h]
  unsigned __int64 v84; // [rsp+80h] [rbp-360h] BYREF
  char v85; // [rsp+88h] [rbp-358h]
  const char *v86; // [rsp+90h] [rbp-350h] BYREF
  __int64 v87; // [rsp+98h] [rbp-348h]
  __int16 v88; // [rsp+B0h] [rbp-330h]
  _QWORD *v89; // [rsp+C0h] [rbp-320h] BYREF
  char v90; // [rsp+C8h] [rbp-318h]
  _QWORD v91[6]; // [rsp+D0h] [rbp-310h] BYREF
  const char *v92; // [rsp+100h] [rbp-2E0h] BYREF
  __int64 v93; // [rsp+108h] [rbp-2D8h]
  __int64 v94; // [rsp+110h] [rbp-2D0h]
  char v95; // [rsp+118h] [rbp-2C8h] BYREF
  char v96; // [rsp+120h] [rbp-2C0h]
  char v97; // [rsp+121h] [rbp-2BFh]
  __int64 v98; // [rsp+1A0h] [rbp-240h] BYREF
  __int64 v99; // [rsp+1A8h] [rbp-238h]
  _BYTE v100[560]; // [rsp+1B0h] [rbp-230h] BYREF

  v5 = a2 + 32;
  if ( !a3 )
    goto LABEL_8;
  sub_9CF0C0((__int64)&v84, a3, a2 + 32, a4);
  v8 = v85 & 1;
  v85 = (2 * (v85 & 1)) | v85 & 0xFD;
  v9 = v84;
  if ( v8 )
    goto LABEL_22;
  v77 = v84;
  if ( !*(_BYTE *)(a2 + 392) )
  {
LABEL_8:
    v13 = *(_DWORD *)(a2 + 68);
    sub_A4DCE0(&v98, v5, 14, 0);
    v9 = v98 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v98 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v14 = v13 + 8;
      v15 = *(_QWORD *)(a2 + 440);
      v68 = v14;
      v98 = (__int64)v100;
      v99 = 0x4000000000LL;
      v89 = v91;
      sub_9C36C0((__int64 *)&v89, *(_BYTE **)(v15 + 232), *(_QWORD *)(v15 + 232) + *(_QWORD *)(v15 + 240));
      v91[2] = *(_QWORD *)(v15 + 264);
      v91[3] = *(_QWORD *)(v15 + 272);
      v17 = *(_QWORD *)(v15 + 280);
      v93 = 0;
      v91[4] = v17;
      v92 = &v95;
      v94 = 128;
      while ( 1 )
      {
        v18 = (__int64 *)v5;
        sub_9CEFB0((__int64)&v82, v5, 0, v16);
        v19 = v83 & 1;
        v20 = (unsigned int)(2 * v19);
        v21 = (2 * v19) | v83 & 0xFD;
        v83 = v21;
        if ( (_BYTE)v19 )
        {
          v18 = (__int64 *)&v82;
          sub_9C9090(a1, (__int64 *)&v82);
          v25 = v83;
          v26 = (v83 & 2) != 0;
          goto LABEL_28;
        }
        if ( (_DWORD)v82 == 1 )
        {
          v25 = v21;
          if ( a3 )
          {
            v18 = (__int64 *)v5;
            sub_9CDFE0((__int64 *)&v86, v5, v77, v20);
            if ( ((unsigned __int64)v86 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              *a1 = (unsigned __int64)v86 & 0xFFFFFFFFFFFFFFFELL | 1;
              v25 = v83;
              v26 = (v83 & 2) != 0;
            }
            else
            {
              v25 = v83;
              *a1 = 1;
              v26 = (v25 & 2) != 0;
            }
            goto LABEL_28;
          }
          *a1 = 1;
          goto LABEL_29;
        }
        if ( (v82 & 0xFFFFFFFD) == 0 )
        {
          v18 = (__int64 *)(a2 + 8);
          v86 = "Malformed block";
          v88 = 259;
          sub_9C81F0(a1, a2 + 8, (__int64)&v86);
          v25 = v83;
          v26 = (v83 & 2) != 0;
          goto LABEL_28;
        }
        LODWORD(v99) = 0;
        sub_A4B600(&v84, v5, HIDWORD(v82), &v98, 0);
        v22 = v85 & 1;
        v16 = (unsigned int)(2 * v22);
        v23 = (2 * v22) | v85 & 0xFD;
        v85 = v23;
        if ( (_BYTE)v22 )
        {
          v18 = (__int64 *)&v84;
          sub_9C8CD0(a1, (__int64 *)&v84);
          goto LABEL_66;
        }
        switch ( (_DWORD)v84 )
        {
          case 2:
            if ( (unsigned __int8)sub_9C3E90(v98, (unsigned int)v99, 1u, &v92)
              || (v43 = *(_QWORD *)(a2 + 1552), v44 = *(unsigned int *)v98, (*(_QWORD *)(a2 + 1560) - v43) >> 3 <= v44)
              || (v45 = *(_QWORD *)(v43 + 8 * v44)) == 0 )
            {
              v18 = (__int64 *)(a2 + 8);
              v86 = "Invalid bbentry record";
              v88 = 259;
              sub_9C81F0(a1, a2 + 8, (__int64)&v86);
LABEL_66:
              if ( (v85 & 2) != 0 )
LABEL_92:
                sub_9CE230(&v84);
              if ( (v85 & 1) != 0 && v84 )
                (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v84 + 8LL))(v84);
              v25 = v83;
              v26 = (v83 & 2) != 0;
LABEL_28:
              if ( v26 )
LABEL_96:
                sub_9CEF10(&v82);
LABEL_29:
              if ( (v25 & 1) != 0 && v82 )
                (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v82 + 8LL))(v82);
              if ( v92 != &v95 )
                _libc_free(v92, v18);
              if ( v89 != v91 )
              {
                v18 = (__int64 *)(v91[0] + 1LL);
                j_j___libc_free_0(v89, v91[0] + 1LL);
              }
              if ( (_BYTE *)v98 != v100 )
                _libc_free(v98, v18);
              return a1;
            }
            v88 = 261;
            v86 = v92;
            v87 = v93;
            sub_BD6B50(v45, &v86);
            v93 = 0;
            break;
          case 3:
            sub_9C8970((__int64)&v86, a2, (__int64 **)&v98, 2u, (__int64)&v89);
            v46 = (v87 & 1) == 0;
            v47 = (unsigned __int64)v86;
            LOBYTE(v87) = v87 & 0xFD;
            if ( v46 )
            {
              if ( *v86 )
                break;
            }
            else
            {
              v86 = 0;
              v81 = v47 | 1;
              if ( (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
                goto LABEL_86;
              v47 = 0;
              if ( MEMORY[0] )
                break;
            }
            v81 = v47;
            v66 = 32 * (*(_QWORD *)(v98 + 8) - 1LL);
            v48 = sub_9DDC30(a2 + 1640, (__int64 *)&v81);
            v16 = v66 + v68;
            *v48 = v16;
            if ( v66 > *(_QWORD *)(a2 + 456) )
              *(_QWORD *)(a2 + 456) = v66;
            if ( (v87 & 2) != 0 )
              sub_9D21E0(&v86);
            if ( (v87 & 1) != 0 && v86 )
              (*(void (__fastcall **)(const char *))(*(_QWORD *)v86 + 8LL))(v86);
            break;
          case 1:
            sub_9C8970((__int64)&v86, a2, (__int64 **)&v98, 1u, (__int64)&v89);
            v41 = v87;
            LOBYTE(v87) = v87 & 0xFD;
            if ( (v41 & 1) != 0 )
            {
              v42 = (unsigned __int64)v86;
              v86 = 0;
              v81 = v42 | 1;
              if ( (v42 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
LABEL_86:
                *a1 = 0;
                v18 = (__int64 *)&v81;
                sub_9C6670(a1, &v81);
                sub_9C66B0((__int64 *)&v81);
                sub_9D2250(&v86);
                goto LABEL_66;
              }
            }
            break;
          default:
            goto LABEL_17;
        }
        v23 = v85;
        if ( (v85 & 2) != 0 )
          goto LABEL_92;
LABEL_17:
        if ( (v23 & 1) != 0 && v84 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v84 + 8LL))(v84);
        if ( (v83 & 2) != 0 )
          goto LABEL_96;
        if ( (v83 & 1) != 0 && v82 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v82 + 8LL))(v82);
      }
    }
LABEL_22:
    *a1 = v9 | 1;
    return a1;
  }
  v10 = *(_DWORD *)(a2 + 68);
  sub_A4DCE0(&v98, v5, 14, 0);
  v12 = v98 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_5:
    *a1 = v12 | 1;
    goto LABEL_6;
  }
  v80 = v10 + 8;
  v98 = (__int64)v100;
  v99 = 0x4000000000LL;
  while ( 1 )
  {
    v27 = (__int64 *)v5;
    sub_9CEFB0((__int64)&v86, v5, 0, v11);
    v28 = v87 & 1;
    v29 = (unsigned int)(2 * v28);
    v30 = (2 * v28) | v87 & 0xFD;
    LOBYTE(v87) = v30;
    if ( (_BYTE)v28 )
    {
      LOBYTE(v87) = v30 & 0xFD;
      v49 = (unsigned __int64)v86;
      v86 = 0;
      v82 = v49 | 1;
      goto LABEL_76;
    }
    if ( (_DWORD)v86 == 1 )
    {
      v82 = 1;
      goto LABEL_76;
    }
    if ( ((unsigned int)v86 & 0xFFFFFFFD) == 0 )
    {
      v27 = (__int64 *)(a2 + 8);
      v97 = 1;
      v92 = "Malformed block";
      v96 = 3;
      sub_9C81F0((__int64 *)&v82, a2 + 8, (__int64)&v92);
      goto LABEL_72;
    }
    LODWORD(v99) = 0;
    sub_A4B600(&v89, v5, HIDWORD(v86), &v98, 0);
    v31 = v90 & 1;
    v11 = (unsigned int)(2 * v31);
    v90 = (2 * v31) | v90 & 0xFD;
    if ( (_BYTE)v31 )
    {
      v27 = (__int64 *)&v89;
      sub_9C8CD0((__int64 *)&v82, (__int64 *)&v89);
      goto LABEL_70;
    }
    if ( (_DWORD)v89 == 3 )
      break;
LABEL_54:
    if ( (v87 & 2) != 0 )
      goto LABEL_104;
    if ( (v87 & 1) != 0 && v86 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v86 + 8LL))(v86);
  }
  v32 = *(_QWORD *)(a2 + 744);
  if ( (unsigned int)*(_QWORD *)v98 < (unsigned int)((*(_QWORD *)(a2 + 752) - v32) >> 5) )
  {
    v11 = *(_QWORD *)(v32 + 32LL * (unsigned int)*(_QWORD *)v98 + 16);
    if ( v11 )
    {
      v33 = *(_DWORD *)(a2 + 1664);
      v34 = 32 * (*(_QWORD *)(v98 + 8) - 1LL);
      v35 = v80 + v34;
      v78 = a2 + 1640;
      if ( v33 )
      {
        v36 = *(_QWORD *)(a2 + 1648);
        v75 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
        v37 = (v33 - 1) & v75;
        v38 = (__int64 *)(v36 + 16LL * v37);
        v39 = *v38;
        if ( *v38 == v11 )
        {
LABEL_48:
          v38[1] = v35;
          if ( v34 > *(_QWORD *)(a2 + 456) )
            *(_QWORD *)(a2 + 456) = v34;
          if ( (v90 & 2) != 0 )
            goto LABEL_109;
          if ( (v90 & 1) != 0 && v89 )
            (*(void (__fastcall **)(_QWORD *))(*v89 + 8LL))(v89);
          goto LABEL_54;
        }
        v69 = 1;
        v72 = 0;
        while ( v39 != -4096 )
        {
          if ( !v72 )
          {
            if ( v39 != -8192 )
              v38 = 0;
            v72 = v38;
          }
          v37 = (v33 - 1) & (v69 + v37);
          v38 = (__int64 *)(v36 + 16LL * v37);
          v39 = *v38;
          if ( v11 == *v38 )
            goto LABEL_48;
          ++v69;
        }
        if ( v72 )
          v38 = v72;
        v50 = *(_DWORD *)(a2 + 1656);
        ++*(_QWORD *)(a2 + 1640);
        v51 = v50 + 1;
        if ( 4 * v51 < 3 * v33 )
        {
          if ( v33 - *(_DWORD *)(a2 + 1660) - v51 <= v33 >> 3 )
          {
            v67 = v80 + v34;
            v71 = v34;
            v74 = v11;
            sub_9DDA50(v78, v33);
            v58 = *(_DWORD *)(a2 + 1664);
            if ( !v58 )
            {
LABEL_157:
              ++*(_DWORD *)(a2 + 1656);
              BUG();
            }
            v59 = v58 - 1;
            v60 = 1;
            v11 = v74;
            v34 = v71;
            v35 = v67;
            v79 = *(_QWORD *)(a2 + 1648);
            v61 = v59 & v75;
            v38 = 0;
            while ( 1 )
            {
              v62 = (__int64 *)(v79 + 16LL * v61);
              v63 = *v62;
              if ( v74 == *v62 )
              {
                v51 = *(_DWORD *)(a2 + 1656) + 1;
                v38 = (__int64 *)(v79 + 16LL * v61);
                goto LABEL_120;
              }
              if ( v63 == -4096 )
                break;
              if ( v63 != -8192 || v38 )
                v62 = v38;
              v65 = v60 + v61;
              v38 = v62;
              ++v60;
              v61 = v59 & v65;
            }
            if ( !v38 )
              v38 = (__int64 *)(v79 + 16LL * v61);
            v51 = *(_DWORD *)(a2 + 1656) + 1;
          }
          goto LABEL_120;
        }
      }
      else
      {
        ++*(_QWORD *)(a2 + 1640);
      }
      v70 = v80 + v34;
      v73 = v34;
      v76 = v11;
      sub_9DDA50(v78, 2 * v33);
      v52 = *(_DWORD *)(a2 + 1664);
      if ( !v52 )
        goto LABEL_157;
      v53 = v52 - 1;
      v11 = v76;
      v54 = 1;
      v34 = v73;
      v35 = v70;
      v55 = 0;
      for ( i = v53 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4)); ; i = v53 & v64 )
      {
        v38 = (__int64 *)(*(_QWORD *)(a2 + 1648) + 16LL * i);
        v57 = *v38;
        if ( v76 == *v38 )
        {
          v51 = *(_DWORD *)(a2 + 1656) + 1;
          goto LABEL_120;
        }
        if ( v57 == -4096 )
          break;
        if ( v57 != -8192 || v55 )
          v38 = v55;
        v64 = v54 + i;
        v55 = v38;
        ++v54;
      }
      if ( v55 )
        v38 = v55;
      v51 = *(_DWORD *)(a2 + 1656) + 1;
LABEL_120:
      *(_DWORD *)(a2 + 1656) = v51;
      if ( *v38 != -4096 )
        --*(_DWORD *)(a2 + 1660);
      *v38 = v11;
      v38[1] = 0;
      goto LABEL_48;
    }
  }
  v27 = (__int64 *)(a2 + 8);
  v97 = 1;
  v92 = "Invalid value reference in symbol table";
  v96 = 3;
  sub_9C81F0((__int64 *)&v82, a2 + 8, (__int64)&v92);
LABEL_70:
  if ( (v90 & 2) != 0 )
LABEL_109:
    sub_9CE230(&v89);
  if ( (v90 & 1) != 0 && v89 )
    (*(void (__fastcall **)(_QWORD *))(*v89 + 8LL))(v89);
LABEL_72:
  if ( (v87 & 2) != 0 )
LABEL_104:
    sub_9CEF10(&v86);
  if ( (v87 & 1) != 0 && v86 )
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v86 + 8LL))(v86);
LABEL_76:
  if ( (_BYTE *)v98 != v100 )
    _libc_free(v98, v27);
  v12 = v82 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v82 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_5;
  sub_9CDFE0(&v98, v5, v77, v29);
  v40 = v98 | 1;
  if ( (v98 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v40 = 1;
  *a1 = v40;
LABEL_6:
  if ( (v85 & 2) != 0 )
    sub_9CDF70(&v84);
  if ( (v85 & 1) != 0 && v84 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v84 + 8LL))(v84);
  return a1;
}
