// Function: sub_2D09560
// Address: 0x2d09560
//
__int64 __fastcall sub_2D09560(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 **v24; // rax
  __int64 **v25; // r13
  char v26; // di
  __int64 **v27; // r12
  __int64 v28; // rsi
  __int64 *v29; // rax
  _BYTE *v30; // r14
  __int64 v31; // rax
  __int64 v32; // rsi
  int v33; // eax
  int v34; // ecx
  unsigned int v35; // eax
  _BYTE *v36; // rdx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rbx
  __int64 *v41; // rax
  __int64 *v42; // r14
  __int64 v43; // r12
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r13
  __int64 v47; // rdx
  __int64 v48; // rcx
  unsigned int v49; // edi
  __int64 *v50; // rax
  __int64 v51; // rsi
  unsigned int v52; // esi
  __int64 v53; // r11
  __int64 v54; // rcx
  unsigned int v55; // edx
  __int64 v56; // rax
  __int64 v57; // rdi
  unsigned int v58; // eax
  unsigned __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  const char *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rax
  int v66; // eax
  unsigned int v67; // r10d
  int v68; // r10d
  int v69; // eax
  int v70; // edx
  __int64 v71; // rcx
  int v72; // edi
  __int64 v73; // rax
  __int64 v74; // [rsp+8h] [rbp-2D8h]
  int v75; // [rsp+18h] [rbp-2C8h]
  char v77; // [rsp+2Eh] [rbp-2B2h]
  unsigned __int8 v78; // [rsp+2Fh] [rbp-2B1h]
  _BYTE *v79; // [rsp+40h] [rbp-2A0h]
  _BYTE *v80; // [rsp+50h] [rbp-290h]
  __int64 v81; // [rsp+58h] [rbp-288h]
  __int64 v82; // [rsp+60h] [rbp-280h]
  __int64 v83; // [rsp+68h] [rbp-278h]
  __int64 v84; // [rsp+70h] [rbp-270h]
  __int64 v85; // [rsp+78h] [rbp-268h]
  unsigned int v86; // [rsp+78h] [rbp-268h]
  unsigned __int8 v87; // [rsp+78h] [rbp-268h]
  __int64 v88; // [rsp+80h] [rbp-260h]
  int v89; // [rsp+88h] [rbp-258h]
  __int64 v92; // [rsp+A8h] [rbp-238h] BYREF
  const char *v93[4]; // [rsp+B0h] [rbp-230h] BYREF
  __int16 v94; // [rsp+D0h] [rbp-210h]
  _BYTE *v95; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v96; // [rsp+E8h] [rbp-1F8h]
  _BYTE v97[64]; // [rsp+F0h] [rbp-1F0h] BYREF
  __int64 v98; // [rsp+130h] [rbp-1B0h] BYREF
  char *v99; // [rsp+138h] [rbp-1A8h]
  __int64 v100; // [rsp+140h] [rbp-1A0h]
  int v101; // [rsp+148h] [rbp-198h]
  char v102; // [rsp+14Ch] [rbp-194h]
  char v103; // [rsp+150h] [rbp-190h] BYREF
  int v104[6]; // [rsp+190h] [rbp-150h] BYREF
  char v105[8]; // [rsp+1A8h] [rbp-138h] BYREF
  unsigned __int64 v106; // [rsp+1B0h] [rbp-130h]
  char v107; // [rsp+1C4h] [rbp-11Ch]
  __int64 v108; // [rsp+210h] [rbp-D0h] BYREF
  __int64 *v109; // [rsp+218h] [rbp-C8h]
  __int64 v110; // [rsp+220h] [rbp-C0h]
  int v111; // [rsp+228h] [rbp-B8h]
  char v112; // [rsp+22Ch] [rbp-B4h]
  _QWORD v113[22]; // [rsp+230h] [rbp-B0h] BYREF

  v79 = *(_BYTE **)a2;
  if ( **(_BYTE **)a2 <= 0x1Cu )
    BUG();
  v88 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 == *(_QWORD *)(*(_QWORD *)a2 + 40LL) )
    return 0;
  sub_2D08230((__int64)v104, a1, a2, *(_QWORD *)a3, 0);
  v78 = sub_2D04F60(v104, a1 + 56);
  if ( !v78 )
    goto LABEL_4;
  v10 = *(_QWORD *)a3;
  v77 = *(_BYTE *)(a3 + 185);
  if ( v77 )
  {
    v73 = sub_AA4FF0(v10);
    if ( v73 )
      v73 -= 24;
    v83 = v73;
    goto LABEL_34;
  }
  v108 = 0;
  v109 = v113;
  v11 = *(_QWORD *)a2;
  v110 = 16;
  v111 = 0;
  v112 = 1;
  v12 = *(_QWORD *)(v11 + 16);
  if ( v12 )
  {
    v13 = 0;
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 24);
      if ( *(_BYTE *)v14 > 0x1Cu )
      {
        if ( *(_BYTE *)v14 == 84 )
        {
          v7 = *(_QWORD *)(v14 - 8);
          v13 = v78;
          v6 = (__int64 *)(32LL * *(unsigned int *)(v14 + 72));
          if ( v10 == *(__int64 *)((char *)&v6[(unsigned int)((v12 - v7) >> 5)] + v7) )
          {
            sub_AE6EC0((__int64)&v108, v14);
            v13 = v78;
          }
        }
        else if ( v10 == *(_QWORD *)(v14 + 40) )
        {
          if ( !v112 )
            goto LABEL_117;
          v15 = v109;
          v7 = HIDWORD(v110);
          v6 = &v109[HIDWORD(v110)];
          if ( v109 != v6 )
          {
            while ( v14 != *v15 )
            {
              if ( v6 == ++v15 )
                goto LABEL_19;
            }
            goto LABEL_11;
          }
LABEL_19:
          if ( HIDWORD(v110) < (unsigned int)v110 )
          {
            v7 = (unsigned int)++HIDWORD(v110);
            *v6 = v14;
            ++v108;
          }
          else
          {
LABEL_117:
            v87 = v13;
            sub_C8CC70((__int64)&v108, v14, (__int64)v6, v7, v13, v8);
            v13 = v87;
          }
        }
      }
LABEL_11:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
      {
        v77 = v13;
        break;
      }
    }
  }
  v16 = *(_QWORD *)(v10 + 56);
  if ( v16 != v10 + 48 )
  {
    v85 = v10;
    v17 = v10 + 48;
    while ( 1 )
    {
      if ( !v16 )
        BUG();
      if ( *(_BYTE *)(v16 - 24) != 84 )
      {
        if ( v112 )
        {
          v18 = v109;
          v19 = &v109[HIDWORD(v110)];
          if ( v109 != v19 )
          {
            while ( v16 - 24 != *v18 )
            {
              if ( v19 == ++v18 )
                goto LABEL_67;
            }
            v83 = *v18;
            goto LABEL_34;
          }
        }
        else if ( sub_C8CA60((__int64)&v108, v16 - 24) )
        {
          v83 = v16 - 24;
          goto LABEL_72;
        }
      }
LABEL_67:
      v16 = *(_QWORD *)(v16 + 8);
      if ( v17 == v16 )
      {
        v10 = v85;
        break;
      }
    }
  }
  v64 = sub_AA4FF0(v10);
  v65 = v64 - 24;
  if ( !v64 )
    v65 = 0;
  v83 = v65;
LABEL_72:
  if ( !v112 )
    _libc_free((unsigned __int64)v109);
LABEL_34:
  v98 = 0;
  v95 = v97;
  v96 = 0x800000000LL;
  v99 = &v103;
  v100 = 8;
  v101 = 0;
  v102 = 1;
  sub_2D04C10((__int64)v79, (__int64)v105, (__int64)&v95, (__int64)&v98);
  v23 = *(unsigned int *)(a3 + 144);
  v111 = 0;
  v112 = 1;
  v109 = v113;
  v110 = 0x100000008LL;
  v108 = 1;
  v113[0] = v88;
  v24 = *(__int64 ***)(a3 + 136);
  v25 = &v24[v23];
  if ( v24 != v25 )
  {
    v26 = v78;
    v27 = *(__int64 ***)(a3 + 136);
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v28 = **v27;
          if ( v26 )
            break;
LABEL_77:
          ++v27;
          sub_C8CC70((__int64)&v108, v28, v23, v20, v21, v22);
          v26 = v112;
          if ( v25 == v27 )
            goto LABEL_42;
        }
        v29 = v109;
        v20 = HIDWORD(v110);
        v23 = (__int64)&v109[HIDWORD(v110)];
        if ( v109 != (__int64 *)v23 )
          break;
LABEL_80:
        if ( HIDWORD(v110) >= (unsigned int)v110 )
          goto LABEL_77;
        v20 = (unsigned int)(HIDWORD(v110) + 1);
        ++v27;
        ++HIDWORD(v110);
        *(_QWORD *)v23 = v28;
        v26 = v112;
        ++v108;
        if ( v25 == v27 )
          goto LABEL_42;
      }
      while ( v28 != *v29 )
      {
        if ( (__int64 *)v23 == ++v29 )
          goto LABEL_80;
      }
      ++v27;
    }
    while ( v25 != v27 );
LABEL_42:
    v89 = v96 - 1;
    if ( (int)v96 - 1 < 0 )
    {
      v84 = 0;
      goto LABEL_108;
    }
    v77 = v78;
LABEL_44:
    v84 = 0;
    v86 = ((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4);
    v82 = 8LL * v89;
    while ( 1 )
    {
      v30 = *(_BYTE **)&v95[v82];
      v31 = *(_QWORD *)(a1 + 48);
      v32 = *(_QWORD *)(v31 + 64);
      v33 = *(_DWORD *)(v31 + 80);
      if ( v33 )
      {
        v34 = v33 - 1;
        v35 = (v33 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v36 = *(_BYTE **)(v32 + 16LL * v35);
        if ( v30 == v36 )
        {
LABEL_47:
          sub_AE6EC0((__int64)a5, *(_QWORD *)&v95[v82]);
        }
        else
        {
          v72 = 1;
          while ( v36 != (_BYTE *)-4096LL )
          {
            v35 = v34 & (v72 + v35);
            v36 = *(_BYTE **)(v32 + 16LL * v35);
            if ( v30 == v36 )
              goto LABEL_47;
            ++v72;
          }
        }
      }
      v37 = sub_B47F80(v30);
      v38 = v84;
      if ( v79 == v30 )
        v38 = v37;
      v84 = v38;
      v39 = 4LL * (*(_DWORD *)(v37 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v37 + 7) & 0x40) != 0 )
      {
        v40 = *(__int64 **)(v37 - 8);
        v41 = &v40[v39];
      }
      else
      {
        v40 = (__int64 *)(v37 - v39 * 8);
        v41 = (__int64 *)v37;
      }
      if ( v41 != v40 )
        break;
LABEL_64:
      v93[0] = sub_BD5D20((__int64)v30);
      v94 = 773;
      v93[1] = v62;
      v93[2] = ".remat";
      sub_BD6B50((unsigned __int8 *)v37, v93);
      v63 = v81;
      LOWORD(v63) = 0;
      v81 = v63;
      sub_B44220((_QWORD *)v37, v83 + 24, v63);
      sub_2D08E40(a1, (__int64)v30, v37, (__int64)&v108);
      --v89;
      v82 -= 8;
      if ( v89 == -1 )
      {
        v26 = v112;
        if ( !v77 )
        {
LABEL_96:
          if ( !v26 )
            _libc_free((unsigned __int64)v109);
          goto LABEL_98;
        }
LABEL_108:
        *a4 = v84;
        goto LABEL_96;
      }
      v83 = v37;
    }
    v80 = v30;
    v42 = v41;
    while ( 1 )
    {
      v43 = *v40;
      if ( !(unsigned __int8)sub_2D04210(*v40) )
      {
LABEL_54:
        v40 += 4;
        if ( v42 == v40 )
          goto LABEL_63;
        continue;
      }
      v46 = *(_QWORD *)(a1 + 48);
      v47 = *(unsigned int *)(v46 + 136);
      v48 = *(_QWORD *)(v46 + 120);
      if ( !(_DWORD)v47 )
        goto LABEL_76;
      v44 = (unsigned int)(v47 - 1);
      v49 = v44 & v86;
      v50 = (__int64 *)(v48 + 16LL * ((unsigned int)v44 & v86));
      v51 = *v50;
      if ( v88 != *v50 )
        break;
LABEL_58:
      v52 = *(_DWORD *)(v46 + 80);
      v53 = v50[1];
      v92 = v43;
      v54 = *(_QWORD *)(v46 + 64);
      if ( v52 )
      {
        v44 = v52 - 1;
        v55 = v44 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v56 = v54 + 16LL * v55;
        v57 = *(_QWORD *)v56;
        if ( v43 != *(_QWORD *)v56 )
        {
          v75 = 1;
          v45 = *(_QWORD *)v56;
          v67 = v44 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          do
          {
            if ( v45 == -4096 )
              goto LABEL_62;
            v67 = v44 & (v75 + v67);
            ++v75;
            v45 = *(_QWORD *)(v54 + 16LL * v67);
          }
          while ( v43 != v45 );
          v56 = v54 + 16LL * ((unsigned int)v44 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4)));
          v68 = 1;
          v45 = 0;
          while ( v57 != -4096 )
          {
            if ( v57 == -8192 && !v45 )
              v45 = v56;
            v55 = v44 & (v68 + v55);
            v56 = v54 + 16LL * v55;
            v57 = *(_QWORD *)v56;
            if ( v43 == *(_QWORD *)v56 )
              goto LABEL_60;
            ++v68;
          }
          v44 = 2 * v52;
          if ( !v45 )
            v45 = v56;
          v93[0] = (const char *)v45;
          v69 = *(_DWORD *)(v46 + 72);
          ++*(_QWORD *)(v46 + 56);
          v70 = v69 + 1;
          if ( 4 * (v69 + 1) >= 3 * v52 )
          {
            v74 = v53;
            v52 *= 2;
          }
          else
          {
            v71 = v43;
            if ( v52 - *(_DWORD *)(v46 + 76) - v70 > v52 >> 3 )
            {
LABEL_92:
              *(_DWORD *)(v46 + 72) = v70;
              if ( *(_QWORD *)v45 != -4096 )
                --*(_DWORD *)(v46 + 76);
              *(_QWORD *)v45 = v71;
              v59 = 0;
              v60 = 0;
              *(_DWORD *)(v45 + 8) = 0;
              goto LABEL_61;
            }
            v74 = v53;
          }
          sub_CE2410(v46 + 56, v52);
          sub_2D064D0(v46 + 56, &v92, v93);
          v71 = v92;
          v45 = (__int64)v93[0];
          v53 = v74;
          v70 = *(_DWORD *)(v46 + 72) + 1;
          goto LABEL_92;
        }
LABEL_60:
        v58 = *(_DWORD *)(v56 + 8);
        v59 = v58 & 0x3F;
        v60 = 8LL * (v58 >> 6);
LABEL_61:
        v54 = *(_QWORD *)(v53 + 24);
        v61 = *(_QWORD *)(v54 + v60);
        if ( _bittest64(&v61, v59) )
          goto LABEL_54;
      }
LABEL_62:
      v40 += 4;
      sub_2D053E0(a1, v43, a5, v54, v44, v45);
      if ( v42 == v40 )
      {
LABEL_63:
        v30 = v80;
        goto LABEL_64;
      }
    }
    v66 = 1;
    while ( v51 != -4096 )
    {
      v45 = (unsigned int)(v66 + 1);
      v49 = v44 & (v66 + v49);
      v50 = (__int64 *)(v48 + 16LL * v49);
      v51 = *v50;
      if ( v88 == *v50 )
        goto LABEL_58;
      v66 = v45;
    }
LABEL_76:
    v50 = (__int64 *)(v48 + 16 * v47);
    goto LABEL_58;
  }
  v89 = v96 - 1;
  if ( (int)v96 - 1 >= 0 )
    goto LABEL_44;
  if ( v77 )
    *a4 = 0;
LABEL_98:
  if ( !v102 )
    _libc_free((unsigned __int64)v99);
  if ( v95 != v97 )
  {
    _libc_free((unsigned __int64)v95);
    if ( v107 )
      return v78;
    goto LABEL_21;
  }
LABEL_4:
  if ( !v107 )
LABEL_21:
    _libc_free(v106);
  return v78;
}
