// Function: sub_2104B30
// Address: 0x2104b30
//
_BOOL8 __fastcall sub_2104B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v9; // rdx
  __int64 *v10; // r13
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 *v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rcx
  char v16; // al
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r14
  int v20; // eax
  char v21; // si
  char v22; // cl
  char v23; // dl
  char *v24; // rax
  size_t v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  int v30; // esi
  char v31; // di
  char v32; // dl
  char v33; // al
  char *v34; // rax
  size_t v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  char v39; // cl
  bool v40; // si
  char v42; // al
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rax
  unsigned int v46; // ebx
  unsigned int v47; // eax
  char v48; // dl
  bool v49; // di
  __int64 v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rsi
  __int64 v53; // r10
  unsigned __int64 v54; // r9
  char v55; // dl
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // esi
  int v59; // eax
  __int64 v60; // rax
  _QWORD *v61; // rax
  unsigned __int64 v62; // [rsp+8h] [rbp-198h]
  __int64 v63; // [rsp+10h] [rbp-190h]
  __int64 v64; // [rsp+18h] [rbp-188h]
  unsigned __int64 v65; // [rsp+20h] [rbp-180h]
  unsigned __int64 v66; // [rsp+20h] [rbp-180h]
  unsigned __int64 v67; // [rsp+20h] [rbp-180h]
  __int64 v68; // [rsp+28h] [rbp-178h]
  __int64 v69; // [rsp+28h] [rbp-178h]
  __int64 v70; // [rsp+28h] [rbp-178h]
  __int64 v71; // [rsp+28h] [rbp-178h]
  __int64 v72; // [rsp+28h] [rbp-178h]
  __int64 v73; // [rsp+28h] [rbp-178h]
  __int64 **v74; // [rsp+38h] [rbp-168h]
  __int64 v75; // [rsp+40h] [rbp-160h]
  unsigned int v76; // [rsp+4Ch] [rbp-154h]
  bool v77; // [rsp+50h] [rbp-150h]
  int v78; // [rsp+50h] [rbp-150h]
  __int64 v79; // [rsp+50h] [rbp-150h]
  __int64 **v80; // [rsp+50h] [rbp-150h]
  __int64 v81; // [rsp+50h] [rbp-150h]
  __int64 v82; // [rsp+58h] [rbp-148h]
  __int64 v83; // [rsp+68h] [rbp-138h]
  __int64 *v84; // [rsp+68h] [rbp-138h]
  __int64 *v85; // [rsp+70h] [rbp-130h]
  __int64 **v86; // [rsp+88h] [rbp-118h]
  _QWORD v87[2]; // [rsp+90h] [rbp-110h] BYREF
  _QWORD v88[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int16 v89; // [rsp+B0h] [rbp-F0h]
  __int64 *v90; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v91; // [rsp+C8h] [rbp-D8h]
  __int64 v92; // [rsp+D0h] [rbp-D0h] BYREF
  const char *v93; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 *v94; // [rsp+E8h] [rbp-B8h]
  __int64 **v95; // [rsp+F0h] [rbp-B0h]
  __int64 v96; // [rsp+F8h] [rbp-A8h]
  const char *v97; // [rsp+100h] [rbp-A0h] BYREF
  const char **v98; // [rsp+108h] [rbp-98h]
  _QWORD v99[2]; // [rsp+110h] [rbp-90h] BYREF
  __int64 *v100; // [rsp+120h] [rbp-80h] BYREF
  __int64 v101; // [rsp+128h] [rbp-78h]
  _BYTE v102[112]; // [rsp+130h] [rbp-70h] BYREF

  v6 = a2 + 8;
  v7 = *(_QWORD *)(a2 + 16);
  v100 = (__int64 *)v102;
  v101 = 0x800000000LL;
  if ( a2 + 8 == v7 )
    return 0;
  do
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      if ( (*(_BYTE *)(v7 - 23) & 0x1C) != 0 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        goto LABEL_9;
    }
    v9 = (unsigned int)v101;
    if ( HIDWORD(v101) == (_DWORD)v101 )
    {
      sub_16CD150((__int64)&v100, v102, (unsigned int)v101 + 1LL, 8, a5, a6);
      v9 = (unsigned int)v101;
    }
    v100[v9] = v7 - 56;
    LODWORD(v101) = v101 + 1;
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v6 != v7 );
LABEL_9:
  v85 = &v100[(unsigned int)v101];
  if ( v100 == v85 )
  {
    v77 = 0;
    goto LABEL_52;
  }
  v10 = v100;
  v77 = 0;
  do
  {
    v11 = *(_QWORD *)a2;
    v12 = *v10;
    v86 = (__int64 **)sub_16471D0(*(_QWORD **)a2, 0);
    v93 = sub_1649960(v12);
    v97 = "__emutls_v.";
    v94 = v13;
    LOWORD(v99[0]) = 1283;
    v98 = &v93;
    sub_16E2FC0((__int64 *)&v90, (__int64)&v97);
    v14 = sub_16321C0(a2, (__int64)v90, v91, 1);
    if ( v14 )
      goto LABEL_11;
    v82 = sub_1632FA0(a2);
    v75 = sub_1599A20(v86);
    if ( sub_15E4F60(v12) )
    {
LABEL_19:
      v84 = (__int64 *)sub_15A9620(v82, v11, 0);
      v18 = (__int64)v86;
      goto LABEL_20;
    }
    v15 = *(_QWORD *)(v12 - 24);
    v16 = *(_BYTE *)(v15 + 16);
    if ( v16 == 13 )
    {
      if ( *(_DWORD *)(v15 + 32) <= 0x40u )
      {
        if ( !*(_QWORD *)(v15 + 24) )
          goto LABEL_19;
      }
      else
      {
        v78 = *(_DWORD *)(v15 + 32);
        v83 = *(_QWORD *)(v12 - 24);
        v17 = sub_16A57B0(v15 + 24);
        v15 = v83;
        if ( v78 == v17 )
          goto LABEL_19;
      }
    }
    else if ( v16 == 10 )
    {
      goto LABEL_19;
    }
    v80 = (__int64 **)v15;
    v84 = (__int64 *)sub_15A9620(v82, v11, 0);
    v18 = sub_1646BA0(*v80, 0);
    v14 = (__int64)v80;
LABEL_20:
    v96 = v18;
    v93 = (const char *)v84;
    v94 = v84;
    v95 = v86;
    v74 = (__int64 **)sub_1644170((__int64 **)&v93, 4);
    v19 = sub_1632210(a2, (__int64)v90, v91, (__int64)v74);
    v20 = *(_BYTE *)(v12 + 32) & 0xF;
    v21 = *(_BYTE *)(v12 + 32) & 0xF;
    if ( (unsigned int)(v20 - 7) > 1 )
    {
      *(_BYTE *)(v19 + 32) = v21 | *(_BYTE *)(v19 + 32) & 0xF0;
    }
    else
    {
      v22 = v21 | *(_BYTE *)(v19 + 32) & 0xC0;
      *(_BYTE *)(v19 + 32) = v22;
      if ( v20 == 7 )
      {
        v23 = *(_BYTE *)(v19 + 33) | 0x40;
        *(_BYTE *)(v19 + 33) = v23;
        *(_BYTE *)(v19 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v22;
LABEL_23:
        *(_BYTE *)(v19 + 33) = v23 | 0x40;
        goto LABEL_24;
      }
    }
    if ( v20 == 8 )
    {
      v23 = *(_BYTE *)(v19 + 33) | 0x40;
      v42 = *(_BYTE *)(v19 + 32) & 0xCF;
      *(_BYTE *)(v19 + 33) = v23;
      *(_BYTE *)(v19 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v42;
      goto LABEL_23;
    }
    v39 = *(_BYTE *)(v19 + 32);
    v40 = v21 != 9;
    if ( (v39 & 0x30) != 0 && v40 )
    {
      v23 = *(_BYTE *)(v19 + 33) | 0x40;
      *(_BYTE *)(v19 + 33) = v23;
      *(_BYTE *)(v19 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v39 & 0xCF;
      if ( v20 == 7 )
        goto LABEL_23;
    }
    else
    {
      *(_BYTE *)(v19 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | *(_BYTE *)(v19 + 32) & 0xCF;
    }
    if ( (*(_BYTE *)(v19 + 32) & 0x30) != 0 && v40 )
    {
      v23 = *(_BYTE *)(v19 + 33);
      goto LABEL_23;
    }
LABEL_24:
    if ( *(_QWORD *)(v12 + 48) )
    {
      v24 = (char *)sub_1649960(v19);
      v26 = sub_1633B90(a2, v24, v25);
      *(_QWORD *)(v19 + 48) = v26;
      *(_DWORD *)(v26 + 8) = *(_DWORD *)(*(_QWORD *)(v12 + 48) + 8LL);
    }
    v77 = sub_15E4F60(v12);
    if ( !v77 )
    {
      v79 = *(_QWORD *)(v12 + 24);
      v76 = (unsigned int)(1 << (*(_DWORD *)(v12 + 32) >> 15)) >> 1;
      if ( !v76 )
        v76 = sub_15A9FE0(v82, v79);
      if ( !v14 )
        goto LABEL_41;
      v87[0] = sub_1649960(v12);
      v89 = 1283;
      v88[0] = "__emutls_t.";
      v87[1] = v27;
      v88[1] = v87;
      sub_16E2FC0((__int64 *)&v97, (__int64)v88);
      v28 = sub_1632210(a2, (__int64)v97, (__int64)v98, v79);
      if ( !v28 || *(_BYTE *)(v28 + 16) != 3 )
      {
        MEMORY[0x50] &= ~1u;
        BUG();
      }
      *(_BYTE *)(v28 + 80) |= 1u;
      v68 = v28;
      sub_15E5440(v28, v14);
      sub_15E4CC0(v68, v76);
      v29 = v68;
      v30 = *(_BYTE *)(v12 + 32) & 0xF;
      v31 = *(_BYTE *)(v12 + 32) & 0xF;
      if ( (unsigned int)(v30 - 7) > 1 )
      {
        *(_BYTE *)(v68 + 32) = v31 | *(_BYTE *)(v68 + 32) & 0xF0;
      }
      else
      {
        v32 = v31 | *(_BYTE *)(v68 + 32) & 0xC0;
        *(_BYTE *)(v68 + 32) = v32;
        if ( v30 == 7 )
        {
          v33 = *(_BYTE *)(v68 + 33) | 0x40;
          *(_BYTE *)(v68 + 33) = v33;
          *(_BYTE *)(v68 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v32;
          goto LABEL_35;
        }
      }
      if ( v30 == 8 )
      {
        v33 = *(_BYTE *)(v68 + 33) | 0x40;
        v55 = *(_BYTE *)(v68 + 32) & 0xCF;
        *(_BYTE *)(v68 + 33) = v33;
        *(_BYTE *)(v68 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v55;
      }
      else
      {
        v48 = *(_BYTE *)(v68 + 32);
        v49 = v31 != 9;
        if ( (v48 & 0x30) == 0 || !v49 )
        {
          *(_BYTE *)(v68 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | *(_BYTE *)(v68 + 32) & 0xCF;
LABEL_71:
          if ( (*(_BYTE *)(v68 + 32) & 0x30) != 0 && v49 )
          {
            v33 = *(_BYTE *)(v68 + 33);
            goto LABEL_35;
          }
LABEL_36:
          if ( *(_QWORD *)(v12 + 48) )
          {
            v34 = (char *)sub_1649960(v68);
            v36 = sub_1633B90(a2, v34, v35);
            v29 = v68;
            *(_QWORD *)(v68 + 48) = v36;
            *(_DWORD *)(v36 + 8) = *(_DWORD *)(*(_QWORD *)(v12 + 48) + 8LL);
          }
          if ( v97 != (const char *)v99 )
          {
            v69 = v29;
            j_j___libc_free_0(v97, v99[0] + 1LL);
            v29 = v69;
          }
          v14 = v29;
LABEL_41:
          v37 = v79;
          v38 = 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v37 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v50 = *(_QWORD *)(v37 + 32);
                v37 = *(_QWORD *)(v37 + 24);
                v38 *= v50;
                continue;
              case 1:
                v43 = 16;
                break;
              case 2:
                v43 = 32;
                break;
              case 3:
              case 9:
                v43 = 64;
                break;
              case 4:
                v43 = 80;
                break;
              case 5:
              case 6:
                v43 = 128;
                break;
              case 7:
                v43 = 8 * (unsigned int)sub_15A9520(v82, 0);
                break;
              case 0xB:
                v43 = *(_DWORD *)(v37 + 8) >> 8;
                break;
              case 0xD:
                v43 = 8LL * *(_QWORD *)sub_15A9930(v82, v37);
                break;
              case 0xE:
                v70 = *(_QWORD *)(v37 + 24);
                v81 = *(_QWORD *)(v37 + 32);
                v51 = sub_15A9FE0(v82, v70);
                v52 = v70;
                v53 = 1;
                v54 = v51;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v52 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v57 = *(_QWORD *)(v52 + 32);
                      v52 = *(_QWORD *)(v52 + 24);
                      v53 *= v57;
                      continue;
                    case 1:
                      v56 = 16;
                      goto LABEL_92;
                    case 2:
                      v56 = 32;
                      goto LABEL_92;
                    case 3:
                    case 9:
                      v56 = 64;
                      goto LABEL_92;
                    case 4:
                      v56 = 80;
                      goto LABEL_92;
                    case 5:
                    case 6:
                      v56 = 128;
                      goto LABEL_92;
                    case 7:
                      v65 = v54;
                      v58 = 0;
                      v71 = v53;
                      goto LABEL_99;
                    case 0xB:
                      v56 = *(_DWORD *)(v52 + 8) >> 8;
                      goto LABEL_92;
                    case 0xD:
                      v67 = v54;
                      v73 = v53;
                      v61 = (_QWORD *)sub_15A9930(v82, v52);
                      v53 = v73;
                      v54 = v67;
                      v56 = 8LL * *v61;
                      goto LABEL_92;
                    case 0xE:
                      v62 = v54;
                      v63 = v53;
                      v72 = *(_QWORD *)(v52 + 32);
                      v64 = *(_QWORD *)(v52 + 24);
                      v66 = (unsigned int)sub_15A9FE0(v82, v64);
                      v60 = sub_127FA20(v82, v64);
                      v53 = v63;
                      v54 = v62;
                      v56 = 8 * v72 * v66 * ((v66 + ((unsigned __int64)(v60 + 7) >> 3) - 1) / v66);
                      goto LABEL_92;
                    case 0xF:
                      v65 = v54;
                      v71 = v53;
                      v58 = *(_DWORD *)(v52 + 8) >> 8;
LABEL_99:
                      v59 = sub_15A9520(v82, v58);
                      v53 = v71;
                      v54 = v65;
                      v56 = (unsigned int)(8 * v59);
LABEL_92:
                      v43 = 8 * v54 * v81 * ((v54 + ((unsigned __int64)(v56 * v53 + 7) >> 3) - 1) / v54);
                      break;
                  }
                  break;
                }
                break;
              case 0xF:
                v43 = 8 * (unsigned int)sub_15A9520(v82, *(_DWORD *)(v37 + 8) >> 8);
                break;
            }
            break;
          }
          v97 = (const char *)sub_159C470((__int64)v84, (unsigned __int64)(v38 * v43 + 7) >> 3, 0);
          v98 = (const char **)sub_159C470((__int64)v84, v76, 0);
          if ( !v14 )
            v14 = v75;
          v99[0] = v75;
          v99[1] = v14;
          v45 = sub_159F090(v74, (__int64 *)&v97, 4, v44);
          sub_15E5440(v19, v45);
          v46 = sub_15A9FE0(v82, (__int64)v86);
          v47 = sub_15A9FE0(v82, (__int64)v84);
          if ( v46 >= v47 )
            v47 = v46;
          sub_15E4CC0(v19, v47);
          v77 = 1;
          goto LABEL_11;
        }
        v33 = *(_BYTE *)(v68 + 33) | 0x40;
        *(_BYTE *)(v68 + 33) = v33;
        *(_BYTE *)(v68 + 32) = *(_BYTE *)(v12 + 32) & 0x30 | v48 & 0xCF;
        if ( v30 != 7 )
          goto LABEL_71;
      }
LABEL_35:
      *(_BYTE *)(v68 + 33) = v33 | 0x40;
      goto LABEL_36;
    }
LABEL_11:
    if ( v90 != &v92 )
      j_j___libc_free_0(v90, v92 + 1);
    ++v10;
  }
  while ( v85 != v10 );
  v85 = v100;
LABEL_52:
  if ( v85 != (__int64 *)v102 )
    _libc_free((unsigned __int64)v85);
  return v77;
}
