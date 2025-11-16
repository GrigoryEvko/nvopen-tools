// Function: sub_322A990
// Address: 0x322a990
//
void __fastcall sub_322A990(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 *v5; // rax
  __int64 v6; // r12
  char *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  char v10; // al
  unsigned int v11; // eax
  unsigned int v12; // ecx
  __int64 *v13; // rsi
  unsigned __int8 v14; // dl
  __int64 v15; // r12
  _QWORD *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // edi
  __int64 *v20; // rcx
  __int64 v21; // rdx
  unsigned __int8 v22; // dl
  int v23; // edx
  __int64 v24; // rcx
  int v25; // edx
  unsigned int v26; // eax
  __int64 *v27; // rdi
  __int64 v28; // rsi
  unsigned __int8 v29; // al
  char *v30; // rdx
  _BYTE *v31; // rsi
  unsigned __int8 v32; // al
  _QWORD *v33; // rsi
  unsigned __int8 v34; // al
  _BYTE **v35; // r14
  __int64 v36; // rax
  unsigned __int8 v37; // dl
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rdi
  void (*v44)(); // r11
  _QWORD *v45; // rdx
  __int64 v46; // r14
  unsigned __int8 v47; // al
  char **v48; // rcx
  char *v49; // r12
  char *v50; // rbx
  char *v51; // rax
  unsigned __int8 v52; // al
  unsigned __int8 v53; // dl
  const char *v54; // rdi
  __int64 v55; // rdx
  unsigned int v56; // r13d
  char v57; // al
  char *v58; // rsi
  __int64 v59; // rdi
  unsigned __int8 v60; // al
  char *v61; // r12
  unsigned int v62; // r12d
  __int64 v63; // r12
  __int64 v64; // rsi
  int v65; // edx
  __int64 v66; // rdi
  int v67; // ecx
  unsigned int v68; // edx
  int v69; // r8d
  bool v70; // cc
  __int64 v71; // rdi
  void (*v72)(); // rax
  unsigned __int8 v73; // al
  __int64 v74; // r12
  __int64 v75; // rdi
  unsigned int v76; // r9d
  int v77; // r8d
  int v78; // edi
  int v79; // r8d
  __int64 v80; // rsi
  int v81; // esi
  int v82; // r10d
  int v83; // r11d
  __int64 *v84; // r10
  int v85; // edx
  unsigned int v86; // eax
  __int64 v87; // rsi
  __int64 *v88; // r8
  int v89; // r8d
  unsigned int v90; // r13d
  __int64 *v91; // rdi
  __int64 v92; // rcx
  unsigned int v93; // [rsp+Ch] [rbp-E4h]
  char v95; // [rsp+17h] [rbp-D9h]
  __int64 v96; // [rsp+18h] [rbp-D8h]
  __int64 v97; // [rsp+20h] [rbp-D0h]
  unsigned int v98; // [rsp+28h] [rbp-C8h]
  unsigned int v99; // [rsp+2Ch] [rbp-C4h]
  const char *v100; // [rsp+40h] [rbp-B0h]
  __int64 v101; // [rsp+48h] [rbp-A8h]
  __int64 v102; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v103; // [rsp+58h] [rbp-98h]
  __int64 v104; // [rsp+60h] [rbp-90h]
  __int64 v105; // [rsp+68h] [rbp-88h]
  _BYTE *v106; // [rsp+70h] [rbp-80h] BYREF
  __int64 v107; // [rsp+78h] [rbp-78h]
  _BYTE v108[112]; // [rsp+80h] [rbp-70h] BYREF

  v106 = v108;
  v107 = 0x800000000LL;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v4 = sub_B10CD0(a2 + 56);
  v5 = (__int64 *)sub_2E88D60(a2);
  v6 = sub_B92180(*v5);
  v7 = (char *)sub_C94E20((__int64)qword_4F863B0);
  if ( v7 )
    v10 = *v7;
  else
    v10 = qword_4F863B0[2];
  v95 = v10 & (v6 != 0);
  if ( v95 )
  {
    v73 = *(_BYTE *)(v6 - 16);
    if ( (v73 & 2) != 0 )
      v74 = *(_QWORD *)(v6 - 32);
    else
      v74 = v6 - 16 - 8LL * ((v73 >> 2) & 0xF);
    v95 = *(_DWORD *)(*(_QWORD *)(v74 + 40) + 32LL) == 3;
  }
  v11 = v107;
  if ( v4 )
  {
    while ( 1 )
    {
LABEL_5:
      if ( (_DWORD)v105 )
      {
        v9 = (unsigned int)(v105 - 1);
        v12 = v9 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v13 = (__int64 *)(v103 + 8LL * v12);
        v8 = *v13;
        if ( *v13 == v4 )
        {
LABEL_7:
          if ( v13 != (__int64 *)(v103 + 8LL * (unsigned int)v105) )
            goto LABEL_18;
        }
        else
        {
          v81 = 1;
          while ( v8 != -4096 )
          {
            v82 = v81 + 1;
            v12 = v9 & (v81 + v12);
            v13 = (__int64 *)(v103 + 8LL * v12);
            v8 = *v13;
            if ( *v13 == v4 )
              goto LABEL_7;
            v81 = v82;
          }
        }
      }
      v14 = *(_BYTE *)(v4 - 16);
      v15 = v4 - 16;
      if ( (v14 & 2) != 0 )
        v16 = *(_QWORD **)(v4 - 32);
      else
        v16 = (_QWORD *)(v15 - 8LL * ((v14 >> 2) & 0xF));
      v17 = v11;
      v18 = v11;
      if ( !*v16 )
        goto LABEL_19;
      if ( (unsigned __int64)v11 + 1 > HIDWORD(v107) )
      {
        sub_C8D5F0((__int64)&v106, v108, v11 + 1LL, 8u, v8, v9);
        v17 = (unsigned int)v107;
      }
      *(_QWORD *)&v106[8 * v17] = v4;
      v11 = v107 + 1;
      LODWORD(v107) = v107 + 1;
      if ( !(_DWORD)v105 )
        break;
      v9 = (unsigned int)(v105 - 1);
      v19 = v9 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v20 = (__int64 *)(v103 + 8LL * v19);
      v21 = *v20;
      if ( *v20 != v4 )
      {
        v83 = 1;
        v84 = 0;
        while ( v21 != -4096 )
        {
          if ( v21 != -8192 || v84 )
            v20 = v84;
          v19 = v9 & (v83 + v19);
          v21 = *(_QWORD *)(v103 + 8LL * v19);
          if ( v21 == v4 )
            goto LABEL_15;
          ++v83;
          v84 = v20;
          v20 = (__int64 *)(v103 + 8LL * v19);
        }
        if ( !v84 )
          v84 = v20;
        ++v102;
        v85 = v104 + 1;
        if ( 4 * ((int)v104 + 1) < (unsigned int)(3 * v105) )
        {
          if ( (int)v105 - HIDWORD(v104) - v85 <= (unsigned int)v105 >> 3 )
          {
            sub_322A7C0((__int64)&v102, v105);
            if ( !(_DWORD)v105 )
            {
LABEL_147:
              LODWORD(v104) = v104 + 1;
              BUG();
            }
            v89 = 1;
            v90 = (v105 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v84 = (__int64 *)(v103 + 8LL * v90);
            v85 = v104 + 1;
            v91 = 0;
            v92 = *v84;
            if ( v4 != *v84 )
            {
              while ( v92 != -4096 )
              {
                if ( !v91 && v92 == -8192 )
                  v91 = v84;
                v9 = (unsigned int)(v89 + 1);
                v90 = (v105 - 1) & (v89 + v90);
                v84 = (__int64 *)(v103 + 8LL * v90);
                v92 = *v84;
                if ( *v84 == v4 )
                  goto LABEL_114;
                ++v89;
              }
              if ( v91 )
                v84 = v91;
            }
          }
          goto LABEL_114;
        }
LABEL_118:
        sub_322A7C0((__int64)&v102, 2 * v105);
        if ( !(_DWORD)v105 )
          goto LABEL_147;
        v86 = (v105 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v84 = (__int64 *)(v103 + 8LL * v86);
        v87 = *v84;
        v85 = v104 + 1;
        if ( *v84 != v4 )
        {
          v9 = 1;
          v88 = 0;
          while ( v87 != -4096 )
          {
            if ( !v88 && v87 == -8192 )
              v88 = v84;
            v86 = (v105 - 1) & (v9 + v86);
            v84 = (__int64 *)(v103 + 8LL * v86);
            v87 = *v84;
            if ( *v84 == v4 )
              goto LABEL_114;
            v9 = (unsigned int)(v9 + 1);
          }
          if ( v88 )
            v84 = v88;
        }
LABEL_114:
        LODWORD(v104) = v85;
        if ( *v84 != -4096 )
          --HIDWORD(v104);
        *v84 = v4;
        v11 = v107;
      }
LABEL_15:
      if ( !v95 )
        goto LABEL_18;
      v22 = *(_BYTE *)(v4 - 16);
      if ( (v22 & 2) != 0 )
      {
        if ( *(_DWORD *)(v4 - 24) != 2 )
          goto LABEL_18;
        v63 = *(_QWORD *)(v4 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
          goto LABEL_18;
        v63 = v15 - 8LL * ((v22 >> 2) & 0xF);
      }
      v64 = *(_QWORD *)(v63 + 8);
      if ( !v64 )
        goto LABEL_18;
      v65 = *(_DWORD *)(a1 + 6256);
      v66 = *(_QWORD *)(a1 + 6240);
      if ( !v65 )
        goto LABEL_18;
      v67 = v65 - 1;
      v68 = (v65 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
      v8 = v68;
      v4 = *(_QWORD *)(v66 + 8LL * v68);
      if ( v64 != v4 )
      {
        v69 = 1;
        while ( v4 != -4096 )
        {
          v9 = (unsigned int)(v69 + 1);
          v68 = v67 & (v69 + v68);
          v8 = v68;
          v4 = *(_QWORD *)(v66 + 8LL * v68);
          if ( v64 == v4 )
            goto LABEL_5;
          v69 = v9;
        }
        goto LABEL_18;
      }
    }
    ++v102;
    goto LABEL_118;
  }
LABEL_18:
  v18 = v11;
LABEL_19:
  v96 = 8LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL) + 1912LL);
  if ( v18 )
  {
    while ( 1 )
    {
      v46 = *(_QWORD *)&v106[8 * v18 - 8];
      LODWORD(v107) = v11 - 1;
      v97 = v46 - 16;
      v47 = *(_BYTE *)(v46 - 16);
      if ( (v47 & 2) != 0 )
      {
        v48 = *(char ***)(v46 - 32);
        v49 = *v48;
        if ( !*v48 )
          goto LABEL_148;
        v50 = 0;
        if ( *(_DWORD *)(v46 - 24) == 2 )
          goto LABEL_73;
      }
      else
      {
        v48 = (char **)(v97 - 8LL * ((v47 >> 2) & 0xF));
        v49 = *v48;
        if ( !*v48 )
LABEL_148:
          JUMPOUT(0x4350E6);
        v50 = 0;
        if ( ((*(_WORD *)(v46 - 16) >> 6) & 0xF) == 2 )
LABEL_73:
          v50 = v48[1];
      }
      v51 = v49;
      if ( *v49 != 16 )
      {
        v52 = *(v49 - 16);
        if ( (v52 & 2) != 0 )
        {
          v51 = (char *)**((_QWORD **)v49 - 4);
          if ( !v51 )
            goto LABEL_75;
        }
        else
        {
          v51 = *(char **)&v49[-8 * ((v52 >> 2) & 0xF) - 16];
          if ( !v51 )
          {
LABEL_75:
            v55 = 0;
            v54 = byte_3F871B3;
            goto LABEL_50;
          }
        }
      }
      v53 = *(v51 - 16);
      if ( (v53 & 2) != 0 )
      {
        v54 = (const char *)**((_QWORD **)v51 - 4);
        if ( v54 )
          goto LABEL_49;
      }
      else
      {
        v54 = *(const char **)&v51[-8 * ((v53 >> 2) & 0xF) - 16];
        if ( v54 )
        {
LABEL_49:
          v54 = (const char *)sub_B91420((__int64)v54);
          goto LABEL_50;
        }
      }
      v55 = 0;
LABEL_50:
      v56 = *(_DWORD *)(v46 + 4);
      v100 = v54;
      v101 = v55;
      v99 = *(unsigned __int16 *)(v46 + 2);
      if ( !v56 )
      {
        v57 = *v49;
LABEL_52:
        v58 = v49;
        v98 = 0;
        v59 = *(_QWORD *)(*(_QWORD *)(a1 + 3232) + v96);
        if ( v57 == 16 )
          goto LABEL_56;
        goto LABEL_53;
      }
      v70 = (unsigned __int16)sub_3220AA0(a1) <= 3u;
      v57 = *v49;
      if ( v70 || v57 != 20 )
        goto LABEL_52;
      v98 = *((_DWORD *)v49 + 1);
      v59 = *(_QWORD *)(*(_QWORD *)(a1 + 3232) + v96);
LABEL_53:
      v60 = *(v49 - 16);
      if ( (v60 & 2) != 0 )
        v61 = (char *)*((_QWORD *)v49 - 4);
      else
        v61 = &v49[-8 * ((v60 >> 2) & 0xF) - 16];
      v58 = *(char **)v61;
LABEL_56:
      v62 = sub_373B2C0(v59, v58);
      if ( LOBYTE(qword_5036408[8]) )
      {
        if ( v56 )
        {
          v71 = *(_QWORD *)(a1 + 8);
          v72 = *(void (**)())(*(_QWORD *)v71 + 312LL);
          if ( v72 != nullsub_1832 )
            ((void (__fastcall *)(__int64, const char *, __int64, _QWORD, _QWORD))v72)(v71, v100, v101, v56, 0);
        }
      }
      if ( !v95 )
        goto LABEL_59;
      v23 = *(_DWORD *)(a1 + 6256);
      v24 = *(_QWORD *)(a1 + 6240);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = v25 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v27 = (__int64 *)(v24 + 8LL * v26);
        v28 = *v27;
        if ( v46 == *v27 )
        {
LABEL_23:
          *v27 = -8192;
          --*(_DWORD *)(a1 + 6248);
          ++*(_DWORD *)(a1 + 6252);
        }
        else
        {
          v75 = *v27;
          v76 = v25 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v77 = 1;
          while ( v75 != -4096 )
          {
            v76 = v25 & (v77 + v76);
            v75 = *(_QWORD *)(v24 + 8LL * v76);
            if ( v46 == v75 )
            {
              v78 = 1;
              if ( v28 == -4096 )
                break;
              while ( 1 )
              {
                v79 = v78 + 1;
                v26 = v25 & (v78 + v26);
                v27 = (__int64 *)(v24 + 8LL * v26);
                v80 = *v27;
                if ( v46 == *v27 )
                  goto LABEL_23;
                v78 = v79;
                if ( v80 == -4096 )
                  goto LABEL_24;
              }
            }
            ++v77;
          }
        }
      }
LABEL_24:
      if ( v50 )
      {
        v29 = *(v50 - 16);
        if ( (v29 & 2) != 0 )
          v30 = (char *)*((_QWORD *)v50 - 4);
        else
          v30 = &v50[-8 * ((v29 >> 2) & 0xF) - 16];
        v31 = *(_BYTE **)v30;
        if ( **(_BYTE **)v30 != 16 )
        {
          v32 = *(v31 - 16);
          if ( (v32 & 2) != 0 )
            v33 = (_QWORD *)*((_QWORD *)v31 - 4);
          else
            v33 = &v31[-8 * ((v32 >> 2) & 0xF) - 16];
          v31 = (_BYTE *)*v33;
        }
        v93 = sub_373B2C0(*(_QWORD *)(*(_QWORD *)(a1 + 3232) + v96), v31);
        v34 = *(_BYTE *)(v46 - 16);
        if ( (v34 & 2) != 0 )
          v35 = *(_BYTE ***)(v46 - 32);
        else
          v35 = (_BYTE **)(v97 - 8LL * ((v34 >> 2) & 0xF));
        v36 = sub_AE7A60(*v35);
        v37 = *(_BYTE *)(v36 - 16);
        if ( (v37 & 2) != 0 )
        {
          v38 = *(_QWORD *)(*(_QWORD *)(v36 - 32) + 24LL);
          if ( !v38 )
            goto LABEL_86;
        }
        else
        {
          v38 = *(_QWORD *)(v36 - 16 - 8LL * ((v37 >> 2) & 0xF) + 24);
          if ( !v38 )
          {
LABEL_86:
            v41 = 0;
            goto LABEL_36;
          }
        }
        v39 = sub_B91420(v38);
        v41 = v40;
        v38 = v39;
LABEL_36:
        v42 = sub_3247180(a1 + 3256, *(_QWORD *)(a1 + 8), v38, v41);
        v43 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
        v44 = *(void (**)())(*(_QWORD *)v43 + 696LL);
        v45 = (_QWORD *)(v42 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v42 & 4) == 0 )
          v45 = (_QWORD *)((v42 & 0xFFFFFFFFFFFFFFF8LL) + 8);
        if ( v44 != nullsub_108 )
          ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64))v44)(
            v43,
            v62,
            v56,
            v99,
            v93,
            *((unsigned int *)v50 + 1),
            *((unsigned __int16 *)v50 + 1),
            *v45,
            a3,
            0,
            v98,
            v100,
            v101);
        v18 = (unsigned int)v107;
        v11 = v107;
        if ( !(_DWORD)v107 )
          break;
      }
      else
      {
LABEL_59:
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 688LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          v62,
          v56,
          v99,
          a3,
          0,
          v98,
          v100,
          v101,
          0,
          0);
        v18 = (unsigned int)v107;
        v11 = v107;
        if ( !(_DWORD)v107 )
          break;
      }
    }
  }
  sub_C7D6A0(v103, 8LL * (unsigned int)v105, 8);
  if ( v106 != v108 )
    _libc_free((unsigned __int64)v106);
}
