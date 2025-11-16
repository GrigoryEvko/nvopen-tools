// Function: sub_13C1940
// Address: 0x13c1940
//
__int64 __fastcall sub_13C1940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  _QWORD *v5; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // edx
  __int64 v12; // rsi
  __int64 v13; // r15
  _QWORD *v14; // rdi
  int v15; // ebx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r15
  __int64 *v22; // rax
  char v23; // dl
  char v24; // r8
  unsigned __int64 v25; // rdx
  char v26; // dl
  __int64 v27; // rdx
  _QWORD *v28; // r8
  _QWORD *v29; // r13
  char v30; // dl
  __int64 v31; // r15
  __int64 *v32; // rax
  __int64 *v33; // rsi
  __int64 *v34; // rcx
  __int64 v35; // rax
  __int64 *v36; // rdi
  unsigned int v37; // r8d
  __int64 *v38; // rdx
  __int64 *v39; // rsi
  __int64 *v40; // rsi
  __int64 *v41; // rcx
  __int64 v42; // rax
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 *v46; // rax
  char v47; // dl
  char v48; // r9
  unsigned __int64 v49; // rdx
  char v50; // dl
  __int64 v51; // rdx
  _QWORD *v52; // r9
  _QWORD *v53; // rsi
  _QWORD *v54; // r12
  _QWORD *v55; // rbx
  char v56; // dl
  __int64 v57; // r14
  __int64 *v58; // rax
  __int64 *v59; // rsi
  __int64 *v60; // rcx
  __int64 v61; // rax
  __int64 *v62; // r9
  unsigned int v63; // edi
  __int64 *v64; // rdx
  __int64 *v65; // rsi
  __int64 *v66; // rsi
  __int64 *v67; // rcx
  __int64 v68; // rax
  __int64 v69; // [rsp+18h] [rbp-1D8h]
  __int64 v70; // [rsp+30h] [rbp-1C0h]
  int v71; // [rsp+38h] [rbp-1B8h]
  __int64 v72; // [rsp+38h] [rbp-1B8h]
  int v73; // [rsp+38h] [rbp-1B8h]
  __int64 v74; // [rsp+38h] [rbp-1B8h]
  _QWORD *v75; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v76; // [rsp+48h] [rbp-1A8h]
  _QWORD v77[8]; // [rsp+50h] [rbp-1A0h] BYREF
  _QWORD *v78; // [rsp+90h] [rbp-160h] BYREF
  __int64 v79; // [rsp+98h] [rbp-158h]
  _QWORD v80[8]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v81; // [rsp+E0h] [rbp-110h] BYREF
  __int64 *v82; // [rsp+E8h] [rbp-108h]
  __int64 *v83; // [rsp+F0h] [rbp-100h]
  __int64 v84; // [rsp+F8h] [rbp-F8h]
  int v85; // [rsp+100h] [rbp-F0h]
  _QWORD v86[9]; // [rsp+108h] [rbp-E8h] BYREF
  __int64 v87; // [rsp+150h] [rbp-A0h] BYREF
  __int64 *v88; // [rsp+158h] [rbp-98h]
  __int64 *v89; // [rsp+160h] [rbp-90h]
  __int64 v90; // [rsp+168h] [rbp-88h]
  int v91; // [rsp+170h] [rbp-80h]
  _QWORD v92[15]; // [rsp+178h] [rbp-78h] BYREF

  v5 = v77;
  v82 = v86;
  v83 = v86;
  v75 = v77;
  v85 = 0;
  v86[0] = a3;
  v81 = 1;
  v77[0] = a3;
  v71 = 0;
  v84 = 0x100000008LL;
  v76 = 0x800000001LL;
  LODWORD(v7) = 1;
  while ( 1 )
  {
    v8 = (unsigned int)v7;
    LODWORD(v7) = v7 - 1;
    v9 = v5[v8 - 1];
    LODWORD(v76) = v7;
    v10 = *(unsigned __int8 *)(v9 + 16);
    if ( (unsigned __int8)v10 <= 3u )
      break;
    if ( (unsigned __int8)(v10 - 17) <= 0x3Du )
    {
      v12 = 0x2000000000001001LL;
      if ( _bittest64(&v12, (unsigned int)(v10 - 17)) )
        goto LABEL_15;
    }
    if ( ++v71 > 4 )
      goto LABEL_21;
    switch ( (_BYTE)v10 )
    {
      case '6':
        v16 = sub_14AD280(*(_QWORD *)(v9 - 24), *(_QWORD *)(a1 + 8), 6);
        v13 = *(_QWORD *)(a1 + 8);
        v14 = v80;
        v88 = v92;
        v90 = 0x100000008LL;
        v89 = v92;
        v78 = v80;
        v91 = 0;
        v87 = 1;
        v92[0] = v16;
        v80[0] = v16;
        v70 = a2;
        v15 = v71;
        v79 = 0x800000001LL;
        LODWORD(v16) = 1;
        do
        {
          v17 = (unsigned int)v16;
          LODWORD(v16) = v16 - 1;
          v18 = v14[v17 - 1];
          LODWORD(v79) = v16;
          v19 = *(unsigned __int8 *)(v18 + 16);
          if ( (unsigned __int8)v19 <= 0x1Du )
          {
            v20 = 537001999;
            if ( _bittest64(&v20, v19) )
              continue;
          }
          else if ( (_BYTE)v19 == 78 )
          {
            continue;
          }
          if ( ++v15 > 4 )
            goto LABEL_32;
          if ( (_BYTE)v19 == 54 )
          {
            v43 = sub_14AD280(*(_QWORD *)(v18 - 24), v13, 6);
            v16 = (unsigned int)v79;
            if ( (unsigned int)v79 >= HIDWORD(v79) )
              goto LABEL_137;
            goto LABEL_93;
          }
          if ( (_BYTE)v19 != 79 )
          {
            if ( (_BYTE)v19 != 77 )
            {
LABEL_32:
              v71 = v15;
              a2 = v70;
              LODWORD(v3) = 0;
              goto LABEL_33;
            }
            v51 = 3LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
            {
              v52 = *(_QWORD **)(v18 - 8);
              v53 = &v52[v51];
            }
            else
            {
              v53 = (_QWORD *)v18;
              v52 = (_QWORD *)(v18 - v51 * 8);
            }
            if ( v52 != v53 )
            {
              v73 = v15;
              v54 = v52;
              v55 = v53;
              v69 = a1;
              while ( 2 )
              {
                v57 = sub_14AD280(*v54, v13, 6);
                v58 = v88;
                if ( v89 != v88 )
                  goto LABEL_104;
                v59 = &v88[HIDWORD(v90)];
                if ( v88 != v59 )
                {
                  v60 = 0;
                  while ( v57 != *v58 )
                  {
                    if ( *v58 == -2 )
                      v60 = v58;
                    if ( v59 == ++v58 )
                    {
                      if ( !v60 )
                        goto LABEL_120;
                      *v60 = v57;
                      --v91;
                      ++v87;
                      goto LABEL_115;
                    }
                  }
                  goto LABEL_105;
                }
LABEL_120:
                if ( HIDWORD(v90) < (unsigned int)v90 )
                {
                  ++HIDWORD(v90);
                  *v59 = v57;
                  ++v87;
LABEL_115:
                  v61 = (unsigned int)v79;
                  if ( (unsigned int)v79 >= HIDWORD(v79) )
                  {
                    sub_16CD150(&v78, v80, 0, 8);
                    v61 = (unsigned int)v79;
                  }
                  v78[v61] = v57;
                  LODWORD(v79) = v79 + 1;
                }
                else
                {
LABEL_104:
                  sub_16CCBA0(&v87, v57);
                  if ( v56 )
                    goto LABEL_115;
                }
LABEL_105:
                v54 += 3;
                if ( v55 == v54 )
                {
                  v15 = v73;
                  a1 = v69;
                  goto LABEL_119;
                }
                continue;
              }
            }
            continue;
          }
          v72 = sub_14AD280(*(_QWORD *)(v18 - 48), v13, 6);
          v44 = sub_14AD280(*(_QWORD *)(v18 - 24), v13, 6);
          v45 = v72;
          v43 = v44;
          v46 = v88;
          if ( v89 != v88 )
            goto LABEL_95;
          v62 = &v88[HIDWORD(v90)];
          v63 = HIDWORD(v90);
          if ( v88 != v62 )
          {
            v64 = v88;
            v65 = 0;
            while ( v72 != *v64 )
            {
              if ( *v64 == -2 )
                v65 = v64;
              if ( v62 == ++v64 )
              {
                if ( !v65 )
                  goto LABEL_151;
                *v65 = v72;
                --v91;
                ++v87;
                goto LABEL_141;
              }
            }
LABEL_128:
            v66 = &v46[v63];
            if ( v66 != v46 )
            {
              v67 = 0;
              do
              {
                if ( v43 == *v46 )
                {
LABEL_119:
                  LODWORD(v16) = v79;
                  goto LABEL_98;
                }
                if ( *v46 == -2 )
                  v67 = v46;
                ++v46;
              }
              while ( v66 != v46 );
              if ( v67 )
              {
                *v67 = v43;
                v16 = (unsigned int)v79;
                --v91;
                ++v87;
                goto LABEL_136;
              }
            }
            if ( v63 < (unsigned int)v90 )
            {
              HIDWORD(v90) = v63 + 1;
              *v66 = v43;
              v16 = (unsigned int)v79;
              ++v87;
LABEL_136:
              if ( HIDWORD(v79) <= (unsigned int)v16 )
              {
LABEL_137:
                sub_16CD150(&v78, v80, 0, 8);
                v16 = (unsigned int)v79;
              }
LABEL_93:
              v78[v16] = v43;
              v14 = v78;
              LODWORD(v16) = v79 + 1;
              LODWORD(v79) = v79 + 1;
              continue;
            }
            goto LABEL_97;
          }
LABEL_151:
          if ( HIDWORD(v90) < (unsigned int)v90 )
          {
            ++HIDWORD(v90);
            *v62 = v72;
            ++v87;
          }
          else
          {
LABEL_95:
            sub_16CCBA0(&v87, v72);
            v46 = v88;
            v45 = v72;
            v48 = v47;
            v49 = (unsigned __int64)v89;
            if ( !v48 )
              goto LABEL_96;
          }
LABEL_141:
          v68 = (unsigned int)v79;
          if ( (unsigned int)v79 >= HIDWORD(v79) )
          {
            v74 = v45;
            sub_16CD150(&v78, v80, 0, 8);
            v68 = (unsigned int)v79;
            v45 = v74;
          }
          v78[v68] = v45;
          v49 = (unsigned __int64)v89;
          LODWORD(v79) = v79 + 1;
          v46 = v88;
LABEL_96:
          if ( (__int64 *)v49 == v46 )
          {
            v63 = HIDWORD(v90);
            goto LABEL_128;
          }
LABEL_97:
          sub_16CCBA0(&v87, v43);
          v16 = (unsigned int)v79;
          if ( v50 )
            goto LABEL_136;
LABEL_98:
          v14 = v78;
        }
        while ( (_DWORD)v16 );
        v71 = v15;
        LODWORD(v3) = 1;
        a2 = v70;
LABEL_33:
        if ( v14 != v80 )
          _libc_free((unsigned __int64)v14);
        if ( v89 != v88 )
          _libc_free((unsigned __int64)v89);
        v5 = v75;
        if ( !(_BYTE)v3 )
          goto LABEL_8;
        LODWORD(v7) = v76;
        if ( !(_DWORD)v76 )
          goto LABEL_16;
        break;
      case 'O':
        v21 = sub_14AD280(*(_QWORD *)(v9 - 48), *(_QWORD *)(a1 + 8), 6);
        v3 = sub_14AD280(*(_QWORD *)(v9 - 24), *(_QWORD *)(a1 + 8), 6);
        v22 = v82;
        if ( v83 == v82 )
        {
          v36 = &v82[HIDWORD(v84)];
          v37 = HIDWORD(v84);
          if ( v82 != v36 )
          {
            v38 = v82;
            v39 = 0;
            while ( v21 != *v38 )
            {
              if ( *v38 == -2 )
                v39 = v38;
              if ( v36 == ++v38 )
              {
                if ( !v39 )
                  goto LABEL_146;
                *v39 = v21;
                --v85;
                ++v81;
                goto LABEL_89;
              }
            }
            goto LABEL_72;
          }
LABEL_146:
          if ( HIDWORD(v84) < (unsigned int)v84 )
          {
            ++HIDWORD(v84);
            *v36 = v21;
            ++v81;
LABEL_89:
            v42 = (unsigned int)v76;
            if ( (unsigned int)v76 >= HIDWORD(v76) )
            {
              sub_16CD150(&v75, v77, 0, 8);
              v42 = (unsigned int)v76;
            }
            v75[v42] = v21;
            v25 = (unsigned __int64)v83;
            LODWORD(v76) = v76 + 1;
            v22 = v82;
            goto LABEL_44;
          }
        }
        sub_16CCBA0(&v81, v21);
        v22 = v82;
        v24 = v23;
        v25 = (unsigned __int64)v83;
        if ( v24 )
          goto LABEL_89;
LABEL_44:
        if ( (__int64 *)v25 != v22 )
          goto LABEL_45;
        v37 = HIDWORD(v84);
LABEL_72:
        v40 = &v22[v37];
        if ( v40 == v22 )
          goto LABEL_144;
        v41 = 0;
        do
        {
          if ( v3 == *v22 )
          {
LABEL_78:
            LODWORD(v7) = v76;
            goto LABEL_46;
          }
          if ( *v22 == -2 )
            v41 = v22;
          ++v22;
        }
        while ( v40 != v22 );
        if ( v41 )
        {
          *v41 = v3;
          v7 = (unsigned int)v76;
          --v85;
          ++v81;
          goto LABEL_84;
        }
LABEL_144:
        if ( v37 >= (unsigned int)v84 )
        {
LABEL_45:
          sub_16CCBA0(&v81, v3);
          v7 = (unsigned int)v76;
          if ( v26 )
            goto LABEL_84;
LABEL_46:
          v5 = v75;
          if ( !(_DWORD)v7 )
            goto LABEL_16;
        }
        else
        {
          HIDWORD(v84) = v37 + 1;
          *v40 = v3;
          v7 = (unsigned int)v76;
          ++v81;
LABEL_84:
          if ( HIDWORD(v76) <= (unsigned int)v7 )
          {
            sub_16CD150(&v75, v77, 0, 8);
            v7 = (unsigned int)v76;
          }
          v75[v7] = v3;
          v5 = v75;
          LODWORD(v7) = v76 + 1;
          LODWORD(v76) = v76 + 1;
LABEL_15:
          if ( !(_DWORD)v7 )
          {
LABEL_16:
            LODWORD(v3) = 1;
            goto LABEL_8;
          }
        }
        break;
      case 'M':
        v27 = 3LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
        {
          v28 = *(_QWORD **)(v9 - 8);
          v3 = (__int64)&v28[v27];
        }
        else
        {
          v3 = v9;
          v28 = (_QWORD *)(v9 - v27 * 8);
        }
        if ( v28 == (_QWORD *)v3 )
          goto LABEL_15;
        v29 = v28;
        while ( 1 )
        {
          v31 = sub_14AD280(*v29, *(_QWORD *)(a1 + 8), 6);
          v32 = v82;
          if ( v83 == v82 )
          {
            v33 = &v82[HIDWORD(v84)];
            if ( v82 != v33 )
            {
              v34 = 0;
              while ( v31 != *v32 )
              {
                if ( *v32 == -2 )
                  v34 = v32;
                if ( v33 == ++v32 )
                {
                  if ( !v34 )
                    goto LABEL_80;
                  *v34 = v31;
                  --v85;
                  ++v81;
                  goto LABEL_63;
                }
              }
              goto LABEL_53;
            }
LABEL_80:
            if ( HIDWORD(v84) < (unsigned int)v84 )
              break;
          }
          sub_16CCBA0(&v81, v31);
          if ( v30 )
            goto LABEL_63;
LABEL_53:
          v29 += 3;
          if ( (_QWORD *)v3 == v29 )
            goto LABEL_78;
        }
        ++HIDWORD(v84);
        *v33 = v31;
        ++v81;
LABEL_63:
        v35 = (unsigned int)v76;
        if ( (unsigned int)v76 >= HIDWORD(v76) )
        {
          sub_16CD150(&v75, v77, 0, 8);
          v35 = (unsigned int)v76;
        }
        v75[v35] = v31;
        LODWORD(v76) = v76 + 1;
        goto LABEL_53;
      default:
        goto LABEL_21;
    }
  }
  if ( v9 == a2 )
  {
LABEL_21:
    LODWORD(v3) = 0;
    goto LABEL_8;
  }
  LOBYTE(v3) = (_BYTE)v10 == 3 && *(_BYTE *)(a2 + 16) == 3;
  if ( (_BYTE)v3 )
  {
    if ( !(unsigned __int8)sub_15E4F60(a2) && !(unsigned __int8)sub_15E4F60(v9) )
      __asm { jmp     rax }
    v5 = v75;
    LODWORD(v3) = 0;
  }
LABEL_8:
  if ( v5 != v77 )
    _libc_free((unsigned __int64)v5);
  if ( v83 != v82 )
    _libc_free((unsigned __int64)v83);
  return (unsigned int)v3;
}
