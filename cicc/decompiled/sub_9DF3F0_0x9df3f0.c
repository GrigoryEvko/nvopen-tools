// Function: sub_9DF3F0
// Address: 0x9df3f0
//
__int64 *__fastcall sub_9DF3F0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rcx
  __int64 *v4; // rsi
  char v5; // dl
  char v6; // al
  char v7; // al
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 *v11; // rax
  const char **v12; // r11
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // esi
  _QWORD *v16; // r8
  unsigned int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // r9
  _DWORD *v20; // rax
  char v21; // cl
  unsigned int v22; // r12d
  char v23; // di
  __int64 v24; // r14
  unsigned int v25; // esi
  unsigned int v27; // eax
  _QWORD *v28; // rcx
  unsigned int v29; // edx
  unsigned int v30; // r8d
  int v31; // r10d
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  int v35; // r12d
  __int64 *v36; // r11
  __int64 v37; // rbx
  int v38; // r13d
  __int64 v39; // rsi
  __int64 *v40; // r10
  int v41; // r8d
  _QWORD *v42; // r9
  unsigned int v43; // ecx
  _QWORD *v44; // rax
  __int64 v45; // r12
  unsigned int v46; // r12d
  unsigned int v47; // ecx
  _QWORD *v48; // rax
  __int64 v49; // rdi
  int v50; // eax
  int v51; // eax
  __int64 v52; // rdi
  int v53; // esi
  _QWORD *v54; // rdi
  unsigned int v55; // edx
  __int64 v56; // r8
  int v57; // r9d
  _QWORD *v58; // rax
  int v59; // esi
  _QWORD *v60; // rdi
  unsigned int v61; // edx
  __int64 v62; // r8
  int v63; // r9d
  __int64 v64; // rbx
  int v65; // esi
  __int64 v66; // r12
  unsigned int v67; // r13d
  __int64 v68; // rcx
  __int64 *v69; // r9
  int v70; // edi
  _QWORD *v71; // r8
  unsigned int v72; // edx
  _QWORD *v73; // rax
  __int64 v74; // r11
  unsigned int v75; // r11d
  unsigned int v76; // edx
  _QWORD *v77; // rax
  __int64 v78; // rsi
  __int64 j; // rax
  __int64 v80; // rax
  int v81; // eax
  int v82; // eax
  __int64 v83; // [rsp+8h] [rbp-4D8h]
  int v84; // [rsp+10h] [rbp-4D0h]
  int v85; // [rsp+10h] [rbp-4D0h]
  __int64 v86; // [rsp+10h] [rbp-4D0h]
  __int64 i; // [rsp+20h] [rbp-4C0h]
  const char **v88; // [rsp+20h] [rbp-4C0h]
  const char **v89; // [rsp+20h] [rbp-4C0h]
  int v90; // [rsp+20h] [rbp-4C0h]
  int v91; // [rsp+20h] [rbp-4C0h]
  __int64 v92; // [rsp+30h] [rbp-4B0h]
  __int64 v95; // [rsp+68h] [rbp-478h] BYREF
  __int64 v96; // [rsp+70h] [rbp-470h] BYREF
  char v97; // [rsp+78h] [rbp-468h]
  __int64 v98; // [rsp+80h] [rbp-460h] BYREF
  char v99; // [rsp+88h] [rbp-458h]
  _QWORD v100[32]; // [rsp+90h] [rbp-450h] BYREF
  const char *v101; // [rsp+190h] [rbp-350h] BYREF
  __int64 v102; // [rsp+198h] [rbp-348h]
  _QWORD *v103; // [rsp+1A0h] [rbp-340h] BYREF
  unsigned int v104; // [rsp+1A8h] [rbp-338h]
  char v105; // [rsp+1B0h] [rbp-330h]
  char v106; // [rsp+1B1h] [rbp-32Fh]
  unsigned __int64 v107; // [rsp+2A0h] [rbp-240h] BYREF
  __int64 v108; // [rsp+2A8h] [rbp-238h]
  _BYTE v109[560]; // [rsp+2B0h] [rbp-230h] BYREF

  v2 = (__int64)(a2 + 4);
  sub_A4DCE0(&v107, a2 + 4, 18, 0);
  if ( (v107 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v107 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v107 = (unsigned __int64)v109;
    v108 = 0x4000000000LL;
    while ( 1 )
    {
      v4 = (__int64 *)v2;
      sub_9CEFB0((__int64)&v96, v2, 0, v3);
      v5 = v97 & 1;
      v6 = (2 * (v97 & 1)) | v97 & 0xFD;
      v97 = v6;
      if ( v5 )
      {
        v97 = v6 & 0xFD;
        v80 = v96;
        v96 = 0;
        *a1 = v80 | 1;
        goto LABEL_34;
      }
      if ( (_DWORD)v96 == 1 )
      {
        *a1 = 1;
        goto LABEL_29;
      }
      if ( (v96 & 0xFFFFFFFD) == 0 )
      {
        v106 = 1;
        v4 = a2 + 1;
        v101 = "Malformed block";
        v105 = 3;
        sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v101);
        goto LABEL_32;
      }
      LODWORD(v108) = 0;
      sub_A4B600(&v98, v2, HIDWORD(v96), &v107, 0);
      v3 = v99 & 1;
      v7 = (2 * v3) | v99 & 0xFD;
      v99 = v7;
      if ( (_BYTE)v3 )
        break;
      if ( (_DWORD)v98 != 1 )
      {
        if ( (_DWORD)v98 != 2 )
          goto LABEL_47;
        LOBYTE(v3) = 1;
      }
      if ( (unsigned int)v108 <= 2 )
      {
        v106 = 1;
        v4 = a2 + 1;
        v101 = "Invalid record";
        v105 = 3;
        sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v101);
        goto LABEL_161;
      }
      v8 = v108 - 1;
      v9 = *(_QWORD *)(v107 + 8LL * (unsigned int)v108 - 8);
      LODWORD(v108) = v108 - 1;
      if ( (_BYTE)v3 )
        v10 = *(_QWORD *)(a2[194] + 8LL * (unsigned int)v9);
      else
        v10 = *(_QWORD *)(a2[93] + 32LL * (unsigned int)v9 + 16);
      v92 = v10;
      v101 = 0;
      v11 = (unsigned __int64 *)&v103;
      v102 = 1;
      do
      {
        *v11 = -4096;
        v11 += 2;
      }
      while ( v11 != &v107 );
      v12 = &v101;
      v13 = *(_QWORD *)(v92 + 16);
      v14 = 0;
      if ( v13 )
      {
        while ( 1 )
        {
          v21 = v102;
          v22 = v14 + 1;
          v23 = v102 & 1;
          if ( (int)v14 + 1 > v8 )
            goto LABEL_44;
          v24 = *(_QWORD *)(v107 + 8 * v14);
          if ( v23 )
          {
            v15 = 15;
            v16 = &v103;
          }
          else
          {
            v25 = v104;
            v16 = v103;
            if ( !v104 )
            {
              v27 = v102;
              ++v101;
              v28 = 0;
              v29 = ((unsigned int)v102 >> 1) + 1;
LABEL_37:
              v30 = 3 * v25;
              goto LABEL_38;
            }
            v15 = v104 - 1;
          }
          v17 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v18 = &v16[2 * v17];
          v19 = *v18;
          if ( v13 == *v18 )
          {
LABEL_19:
            v20 = v18 + 1;
            goto LABEL_20;
          }
          v31 = 1;
          v28 = 0;
          while ( v19 != -4096 )
          {
            if ( !v28 && v19 == -8192 )
              v28 = v18;
            v17 = v15 & (v31 + v17);
            v18 = &v16[2 * v17];
            v19 = *v18;
            if ( *v18 == v13 )
              goto LABEL_19;
            ++v31;
          }
          v30 = 48;
          v25 = 16;
          if ( !v28 )
            v28 = v18;
          v27 = v102;
          ++v101;
          v29 = ((unsigned int)v102 >> 1) + 1;
          if ( !v23 )
          {
            v25 = v104;
            goto LABEL_37;
          }
LABEL_38:
          if ( v30 <= 4 * v29 )
          {
            v88 = v12;
            sub_9DEFB0((__int64)v12, 2 * v25);
            v12 = v88;
            if ( (v102 & 1) != 0 )
            {
              v53 = 15;
              v54 = &v103;
            }
            else
            {
              v54 = v103;
              if ( !v104 )
                goto LABEL_174;
              v53 = v104 - 1;
            }
            v27 = v102;
            v55 = v53 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v28 = &v54[2 * v55];
            v56 = *v28;
            if ( *v28 == v13 )
              goto LABEL_40;
            v57 = 1;
            v58 = 0;
            while ( v56 != -4096 )
            {
              if ( v56 == -8192 && !v58 )
                v58 = v28;
              v55 = v53 & (v57 + v55);
              v28 = &v54[2 * v55];
              v56 = *v28;
              if ( *v28 == v13 )
                goto LABEL_105;
              ++v57;
            }
          }
          else
          {
            if ( v25 - HIDWORD(v102) - v29 > v25 >> 3 )
              goto LABEL_40;
            v89 = v12;
            sub_9DEFB0((__int64)v12, v25);
            v12 = v89;
            if ( (v102 & 1) != 0 )
            {
              v59 = 15;
              v60 = &v103;
            }
            else
            {
              v60 = v103;
              if ( !v104 )
              {
LABEL_174:
                LODWORD(v102) = (2 * ((unsigned int)v102 >> 1) + 2) | v102 & 1;
                BUG();
              }
              v59 = v104 - 1;
            }
            v27 = v102;
            v61 = v59 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v28 = &v60[2 * v61];
            v62 = *v28;
            if ( *v28 == v13 )
              goto LABEL_40;
            v63 = 1;
            v58 = 0;
            while ( v62 != -4096 )
            {
              if ( !v58 && v62 == -8192 )
                v58 = v28;
              v61 = v59 & (v63 + v61);
              v28 = &v60[2 * v61];
              v62 = *v28;
              if ( *v28 == v13 )
                goto LABEL_105;
              ++v63;
            }
          }
          if ( v58 )
            v28 = v58;
LABEL_105:
          v27 = v102;
LABEL_40:
          LODWORD(v102) = (2 * (v27 >> 1) + 2) | v27 & 1;
          if ( *v28 != -4096 )
            --HIDWORD(v102);
          *v28 = v13;
          v20 = v28 + 1;
          *((_DWORD *)v28 + 2) = 0;
LABEL_20:
          *v20 = v24;
          v13 = *(_QWORD *)(v13 + 8);
          v8 = v108;
          if ( !v13 )
          {
            v21 = v102;
            if ( (_DWORD)v108 != (unsigned int)v102 >> 1 )
              goto LABEL_44;
            if ( v22 > (unsigned int)v108 )
              goto LABEL_44;
            v32 = *(_QWORD *)(v92 + 16);
            if ( !v32 )
              goto LABEL_44;
            v33 = *(_QWORD *)(v32 + 8);
            if ( !v33 )
              goto LABEL_44;
            *(_QWORD *)(v32 + 8) = 0;
            v34 = *(_QWORD *)(v33 + 8);
            v100[0] = *(_QWORD *)(v92 + 16);
            if ( v34 )
            {
              v83 = v2;
              v35 = 1;
              for ( i = v34; ; i = *(_QWORD *)(i + 8) )
              {
                *(_QWORD *)(v33 + 8) = 0;
                v36 = v100;
                v37 = 0;
                v38 = v35;
                while ( 1 )
                {
                  v39 = *v36;
                  if ( !*v36 )
                    break;
                  v40 = &v95;
                  while ( 1 )
                  {
                    if ( !v33 )
                    {
LABEL_78:
                      *v40 = v39;
                      goto LABEL_79;
                    }
                    while ( 1 )
                    {
                      if ( (v102 & 1) != 0 )
                      {
                        v41 = 15;
                        v42 = &v103;
                      }
                      else
                      {
                        v42 = v103;
                        if ( !v104 )
                          break;
                        v41 = v104 - 1;
                      }
                      v43 = v41 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
                      v44 = &v42[2 * v43];
                      v45 = *v44;
                      if ( *v44 == v33 )
                      {
LABEL_73:
                        v46 = *((_DWORD *)v44 + 2);
                      }
                      else
                      {
                        v50 = 1;
                        while ( v45 != -4096 )
                        {
                          v43 = v41 & (v50 + v43);
                          v85 = v50 + 1;
                          v44 = &v42[2 * v43];
                          v45 = *v44;
                          if ( *v44 == v33 )
                            goto LABEL_73;
                          v50 = v85;
                        }
                        v46 = 0;
                      }
                      if ( (v102 & 1) == 0 && !v104 )
                        break;
                      v47 = v41 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                      v48 = &v42[2 * v47];
                      v49 = *v48;
                      if ( v39 != *v48 )
                      {
                        v51 = 1;
                        if ( v49 == -4096 )
                          break;
                        while ( 1 )
                        {
                          v47 = v41 & (v51 + v47);
                          v84 = v51 + 1;
                          v48 = &v42[2 * v47];
                          v52 = *v48;
                          if ( *v48 == v39 )
                            break;
                          v51 = v84;
                          if ( v52 == -4096 )
                            goto LABEL_87;
                        }
                      }
                      if ( *((_DWORD *)v48 + 2) <= v46 )
                        break;
                      *v40 = v33;
                      v40 = (__int64 *)(v33 + 8);
                      v33 = *(_QWORD *)(v33 + 8);
                      if ( !v33 )
                        goto LABEL_78;
                    }
LABEL_87:
                    *v40 = v39;
                    v40 = (__int64 *)(v39 + 8);
                    if ( !*(_QWORD *)(v39 + 8) )
                      break;
                    v39 = *(_QWORD *)(v39 + 8);
                  }
                  *(_QWORD *)(v39 + 8) = v33;
LABEL_79:
                  v37 = (unsigned int)(v37 + 1);
                  *v36 = 0;
                  v33 = v95;
                  ++v36;
                  if ( v38 == (_DWORD)v37 )
                  {
                    v35 = v38 + 1;
                    goto LABEL_81;
                  }
                }
                v35 = v38;
LABEL_81:
                v100[v37] = v33;
                v33 = i;
                if ( !*(_QWORD *)(i + 8) )
                  break;
              }
              v64 = i;
              v65 = v35;
              v2 = v83;
              *(_QWORD *)(v92 + 16) = i;
            }
            else
            {
              v64 = v33;
              v65 = 1;
              *(_QWORD *)(v92 + 16) = v33;
            }
            v86 = v2;
            v66 = 0;
            v67 = v65;
            do
            {
              v68 = v100[v66];
              v69 = &v95;
              if ( v68 )
              {
                while ( 1 )
                {
                  if ( !v64 )
                  {
LABEL_135:
                    *v69 = v68;
                    goto LABEL_136;
                  }
                  while ( 1 )
                  {
                    if ( (v102 & 1) != 0 )
                    {
                      v70 = 15;
                      v71 = &v103;
                    }
                    else
                    {
                      v71 = v103;
                      if ( !v104 )
                        break;
                      v70 = v104 - 1;
                    }
                    v72 = v70 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                    v73 = &v71[2 * v72];
                    v74 = *v73;
                    if ( v64 == *v73 )
                    {
LABEL_130:
                      v75 = *((_DWORD *)v73 + 2);
                    }
                    else
                    {
                      v81 = 1;
                      while ( v74 != -4096 )
                      {
                        v72 = v70 & (v81 + v72);
                        v91 = v81 + 1;
                        v73 = &v71[2 * v72];
                        v74 = *v73;
                        if ( v64 == *v73 )
                          goto LABEL_130;
                        v81 = v91;
                      }
                      v75 = 0;
                    }
                    if ( (v102 & 1) == 0 && !v104 )
                      break;
                    v76 = v70 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
                    v77 = &v71[2 * v76];
                    v78 = *v77;
                    if ( *v77 != v68 )
                    {
                      v82 = 1;
                      while ( v78 != -4096 )
                      {
                        v76 = v70 & (v82 + v76);
                        v90 = v82 + 1;
                        v77 = &v71[2 * v76];
                        v78 = *v77;
                        if ( v68 == *v77 )
                          goto LABEL_133;
                        v82 = v90;
                      }
                      break;
                    }
LABEL_133:
                    if ( *((_DWORD *)v77 + 2) <= v75 )
                      break;
                    *v69 = v64;
                    v69 = (__int64 *)(v64 + 8);
                    v64 = *(_QWORD *)(v64 + 8);
                    if ( !v64 )
                      goto LABEL_135;
                  }
                  *v69 = v68;
                  v69 = (__int64 *)(v68 + 8);
                  if ( !*(_QWORD *)(v68 + 8) )
                    break;
                  v68 = *(_QWORD *)(v68 + 8);
                }
                *(_QWORD *)(v68 + 8) = v64;
LABEL_136:
                v64 = v95;
                *(_QWORD *)(v92 + 16) = v95;
              }
              ++v66;
            }
            while ( v67 > (unsigned int)v66 );
            v2 = v86;
            for ( j = v92 + 16; v64; v64 = *(_QWORD *)(v64 + 8) )
            {
              *(_QWORD *)(v64 + 16) = j;
              j = v64 + 8;
            }
            break;
          }
          v14 = v22;
        }
      }
      v21 = v102;
LABEL_44:
      v3 = v21 & 1;
      if ( !(_DWORD)v3 )
        sub_C7D6A0(v103, 16LL * v104, 8);
      v7 = v99;
      if ( (v99 & 2) != 0 )
LABEL_83:
        sub_9CE230(&v98);
LABEL_47:
      if ( (v7 & 1) != 0 && v98 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v98 + 8LL))(v98);
      if ( (v97 & 2) != 0 )
LABEL_97:
        sub_9CEF10(&v96);
      if ( (v97 & 1) != 0 )
      {
        if ( v96 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v96 + 8LL))(v96);
      }
    }
    v4 = &v98;
    sub_9C8CD0(a1, &v98);
LABEL_161:
    if ( (v99 & 2) != 0 )
      goto LABEL_83;
    if ( (v99 & 1) != 0 && v98 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v98 + 8LL))(v98);
LABEL_32:
    if ( (v97 & 2) != 0 )
      goto LABEL_97;
    if ( (v97 & 1) != 0 )
    {
LABEL_34:
      if ( v96 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v96 + 8LL))(v96);
    }
LABEL_29:
    if ( (_BYTE *)v107 != v109 )
      _libc_free(v107, v4);
  }
  return a1;
}
