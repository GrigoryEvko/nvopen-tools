// Function: sub_105B420
// Address: 0x105b420
//
__int64 *__fastcall sub_105B420(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 *result; // rax
  _QWORD *v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 *v19; // r13
  __int64 *v20; // r12
  __int64 *v21; // r12
  __int64 v22; // r14
  int v23; // ecx
  _QWORD *v24; // rdi
  _QWORD *v25; // rsi
  __int64 v26; // r8
  __int64 v27; // rbx
  unsigned int v28; // r9d
  _QWORD *v29; // rdi
  _QWORD *v30; // rsi
  int v31; // edx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 *v34; // r10
  int v35; // eax
  unsigned int v36; // esi
  __int64 *v37; // rcx
  __int64 v38; // rdi
  __int64 v39; // r9
  __int64 v40; // r8
  char v41; // al
  _QWORD *v42; // rdi
  _QWORD *v43; // rsi
  int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // rdx
  int v47; // eax
  unsigned int v48; // esi
  __int64 *v49; // rcx
  __int64 v50; // rdi
  char v51; // al
  __int64 v52; // r12
  char v53; // al
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 *v56; // rax
  char *v57; // r12
  __int64 v58; // rbx
  char *v59; // r13
  unsigned __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 **v63; // r14
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 **v68; // r12
  __int64 v69; // rdx
  __int64 v70; // rsi
  __int64 *v71; // rdi
  int v72; // eax
  unsigned int v73; // ecx
  __int64 *v74; // rdx
  __int64 v75; // r9
  char v76; // al
  int v77; // ecx
  int v78; // r10d
  __int64 v79; // rcx
  int v80; // ecx
  int v81; // r11d
  __int64 v82; // rcx
  char *v83; // rbx
  __int64 v84; // rcx
  __int64 v85; // rdx
  char *v86; // rax
  char *v87; // rsi
  __int64 v88; // rax
  unsigned __int64 v89; // rcx
  __int64 v90; // r12
  __int64 v91; // r14
  __int64 v92; // rbx
  unsigned __int8 *v93; // rsi
  __int64 *v94; // rax
  __int64 *v95; // rdx
  __int64 *v96; // rax
  char *v97; // rsi
  int v98; // edx
  int v99; // r8d
  __int64 *v100; // [rsp+8h] [rbp-C8h]
  _QWORD *v101; // [rsp+18h] [rbp-B8h]
  __int64 v102; // [rsp+20h] [rbp-B0h]
  unsigned int v103; // [rsp+2Ch] [rbp-A4h]
  __int64 *v104; // [rsp+30h] [rbp-A0h]
  char *v105; // [rsp+30h] [rbp-A0h]
  __int64 *v107; // [rsp+38h] [rbp-98h]
  __int64 *v108; // [rsp+40h] [rbp-90h]
  __int64 v109; // [rsp+40h] [rbp-90h]
  __int64 v110; // [rsp+48h] [rbp-88h]
  __int64 **v111; // [rsp+48h] [rbp-88h]
  __int64 v112; // [rsp+58h] [rbp-78h] BYREF
  void *src; // [rsp+60h] [rbp-70h] BYREF
  __int64 v114; // [rsp+68h] [rbp-68h]
  _BYTE v115[96]; // [rsp+70h] [rbp-60h] BYREF

  v6 = a1;
  v7 = *(_QWORD *)(a2 + 40);
  v102 = v7;
  if ( !*(_BYTE *)(a1 + 300) )
    goto LABEL_24;
  v8 = *(__int64 **)(a1 + 280);
  a4 = *(unsigned int *)(a1 + 292);
  a3 = &v8[a4];
  if ( v8 == a3 )
  {
LABEL_23:
    if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 288) )
    {
LABEL_24:
      sub_C8CC70(a1 + 272, v7, (__int64)a3, a4, a5, a6);
      goto LABEL_6;
    }
    *(_DWORD *)(a1 + 292) = a4 + 1;
    *a3 = v7;
    ++*(_QWORD *)(a1 + 272);
  }
  else
  {
    while ( v7 != *v8 )
    {
      if ( a3 == ++v8 )
        goto LABEL_23;
    }
  }
LABEL_6:
  v9 = *(_QWORD *)(a1 + 584);
  if ( v7 )
  {
    v10 = (__int64 *)(unsigned int)(*(_DWORD *)(v7 + 44) + 1);
    result = v10;
  }
  else
  {
    v10 = 0;
    result = 0;
  }
  if ( (unsigned int)result < *(_DWORD *)(v9 + 32) )
  {
    result = *(__int64 **)(v9 + 24);
    if ( result[(_QWORD)v10] )
    {
      src = v115;
      v114 = 0x600000000LL;
      v12 = sub_105AE00(a1 + 816, v7);
      v101 = v12;
      v13 = (__int64 *)v12[1];
      if ( *((_BYTE *)v12 + 28) )
        v14 = *((unsigned int *)v12 + 5);
      else
        v14 = *((unsigned int *)v12 + 4);
      v108 = &v13[v14];
      if ( v13 != v108 )
      {
        while ( 1 )
        {
          v15 = *v13;
          if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v108 == ++v13 )
            goto LABEL_15;
        }
        v104 = v13;
        if ( v108 != v13 )
        {
          v21 = &v112;
          v103 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
          while ( 1 )
          {
            v22 = sub_E387E0(*(_QWORD *)(a1 + 224), v15);
            v110 = *(_QWORD *)(a1 + 584);
            if ( !v22 )
              goto LABEL_86;
            v23 = *(_DWORD *)(v22 + 72);
            v112 = v102;
            if ( v23 )
            {
              v69 = *(unsigned int *)(v22 + 80);
              v70 = *(_QWORD *)(v22 + 64);
              v71 = (__int64 *)(v70 + 8 * v69);
              if ( !(_DWORD)v69 )
                goto LABEL_36;
              v72 = v69 - 1;
              v73 = (v69 - 1) & v103;
              v74 = (__int64 *)(v70 + 8LL * v73);
              v75 = *v74;
              if ( v102 != *v74 )
              {
                v98 = 1;
                while ( v75 != -4096 )
                {
                  v99 = v98 + 1;
                  v73 = v72 & (v98 + v73);
                  v74 = (__int64 *)(v70 + 8LL * v73);
                  v75 = *v74;
                  if ( v102 == *v74 )
                    goto LABEL_89;
                  v98 = v99;
                }
LABEL_36:
                v26 = *(_QWORD *)v22;
                v27 = v22;
                if ( *(_QWORD *)v22 )
                {
                  v28 = ((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4);
                  while ( 1 )
                  {
                    v31 = *(_DWORD *)(v26 + 72);
                    v112 = v102;
                    if ( v31 )
                    {
                      v32 = *(unsigned int *)(v26 + 80);
                      v33 = *(_QWORD *)(v26 + 64);
                      v34 = (__int64 *)(v33 + 8 * v32);
                      if ( (_DWORD)v32 )
                      {
                        v35 = v32 - 1;
                        v36 = (v32 - 1) & v28;
                        v37 = (__int64 *)(v33 + 8LL * v36);
                        v38 = *v37;
                        if ( v102 == *v37 )
                        {
LABEL_44:
                          if ( v37 != v34 )
                            break;
                        }
                        else
                        {
                          v80 = 1;
                          while ( v38 != -4096 )
                          {
                            v81 = v80 + 1;
                            v82 = v35 & (v36 + v80);
                            v36 = v82;
                            v37 = (__int64 *)(v33 + 8 * v82);
                            v38 = *v37;
                            if ( v102 == *v37 )
                              goto LABEL_44;
                            v80 = v81;
                          }
                        }
                      }
                    }
                    else
                    {
                      v29 = *(_QWORD **)(v26 + 88);
                      v30 = &v29[*(unsigned int *)(v26 + 96)];
                      if ( v30 != sub_1055FB0(v29, (__int64)v30, v21) )
                        break;
                    }
                    v27 = v26;
                    if ( !*(_QWORD *)v26 )
                      break;
                    v26 = *(_QWORD *)v26;
                  }
                }
                if ( *(_DWORD *)(v27 + 16) != 1 )
                {
                  sub_B196A0(v110, v102, v15);
                  v40 = v103;
                  if ( !v41 )
                    goto LABEL_50;
                  goto LABEL_63;
                }
                goto LABEL_90;
              }
LABEL_89:
              if ( v71 == v74 )
                goto LABEL_36;
            }
            else
            {
              v24 = *(_QWORD **)(v22 + 88);
              v25 = &v24[*(unsigned int *)(v22 + 96)];
              if ( v25 == sub_1055FB0(v24, (__int64)v25, v21) )
                goto LABEL_36;
            }
LABEL_90:
            sub_B196A0(v110, v102, v15);
            if ( v76 )
              goto LABEL_86;
            v40 = v103;
            v27 = 0;
            while ( 1 )
            {
LABEL_50:
              v44 = *(_DWORD *)(v22 + 72);
              v112 = v102;
              if ( !v44 )
              {
                v42 = *(_QWORD **)(v22 + 88);
                v43 = &v42[*(unsigned int *)(v22 + 96)];
                if ( v43 != sub_1055FB0(v42, (__int64)v43, v21) )
                  break;
                goto LABEL_49;
              }
              v45 = *(unsigned int *)(v22 + 80);
              v46 = *(_QWORD *)(v22 + 64);
              v39 = v46 + 8 * v45;
              if ( (_DWORD)v45 )
              {
                v47 = v45 - 1;
                v48 = (v45 - 1) & v40;
                v49 = (__int64 *)(v46 + 8LL * v48);
                v50 = *v49;
                if ( v102 != *v49 )
                {
                  v77 = 1;
                  while ( v50 != -4096 )
                  {
                    v78 = v77 + 1;
                    v79 = v47 & (v48 + v77);
                    v48 = v79;
                    v49 = (__int64 *)(v46 + 8 * v79);
                    v50 = *v49;
                    if ( v102 == *v49 )
                      goto LABEL_53;
                    v77 = v78;
                  }
                  goto LABEL_49;
                }
LABEL_53:
                if ( v49 != (__int64 *)v39 )
                  break;
              }
LABEL_49:
              v22 = *(_QWORD *)v22;
              if ( !v22 )
                goto LABEL_85;
            }
            if ( *(_DWORD *)(v22 + 16) == 1 || (sub_B196A0(v110, **(_QWORD **)(v22 + 8), v15), v51) )
            {
LABEL_85:
              if ( v27 )
                goto LABEL_63;
LABEL_86:
              sub_1058130(a1, v15);
              goto LABEL_65;
            }
            if ( !*(_QWORD *)v22 )
              goto LABEL_62;
            v100 = v21;
            v52 = *(_QWORD *)v22;
            while ( 2 )
            {
              sub_B196A0(v110, **(_QWORD **)(v52 + 8), v15);
              if ( v53 )
              {
                v21 = v100;
LABEL_62:
                v27 = v22;
LABEL_63:
                v54 = (unsigned int)v114;
                v55 = (unsigned int)v114 + 1LL;
                if ( v55 > HIDWORD(v114) )
                  goto LABEL_93;
                goto LABEL_64;
              }
              v22 = v52;
              if ( *(_QWORD *)v52 )
              {
                v52 = *(_QWORD *)v52;
                continue;
              }
              break;
            }
            v54 = (unsigned int)v114;
            v27 = v52;
            v21 = v100;
            v55 = (unsigned int)v114 + 1LL;
            if ( v55 > HIDWORD(v114) )
            {
LABEL_93:
              sub_C8D5F0((__int64)&src, v115, v55, 8u, v40, v39);
              v54 = (unsigned int)v114;
            }
LABEL_64:
            *((_QWORD *)src + v54) = v27;
            LODWORD(v114) = v114 + 1;
LABEL_65:
            v56 = v104 + 1;
            if ( v104 + 1 != v108 )
            {
              while ( 1 )
              {
                v15 = *v56;
                if ( (unsigned __int64)*v56 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v108 == ++v56 )
                  goto LABEL_68;
              }
              v104 = v56;
              if ( v108 != v56 )
                continue;
            }
LABEL_68:
            v57 = (char *)src;
            v6 = a1;
            v58 = 8LL * (unsigned int)v114;
            v59 = (char *)src + v58;
            if ( src == (char *)src + v58 )
              break;
            _BitScanReverse64(&v60, v58 >> 3);
            sub_1058BC0((char *)src, (__int64 *)((char *)src + v58), 2LL * (int)(63 - (v60 ^ 0x3F)));
            if ( (unsigned __int64)v58 > 0x80 )
            {
              v83 = v57 + 128;
              sub_1055EE0(v57, v57 + 128);
              if ( v57 + 128 != v59 )
              {
                do
                {
                  while ( 1 )
                  {
                    v84 = *(_QWORD *)v83;
                    v85 = *((_QWORD *)v83 - 1);
                    v86 = v83 - 8;
                    if ( *(_DWORD *)(v85 + 168) < *(_DWORD *)(*(_QWORD *)v83 + 168LL) )
                      break;
                    v97 = v83;
                    v83 += 8;
                    *(_QWORD *)v97 = v84;
                    if ( v83 == v59 )
                      goto LABEL_71;
                  }
                  do
                  {
                    *((_QWORD *)v86 + 1) = v85;
                    v87 = v86;
                    v85 = *((_QWORD *)v86 - 1);
                    v86 -= 8;
                  }
                  while ( *(_DWORD *)(v84 + 168) > *(_DWORD *)(v85 + 168) );
                  v83 += 8;
                  *(_QWORD *)v87 = v84;
                }
                while ( v83 != v59 );
              }
            }
            else
            {
              sub_1055EE0(v57, v59);
            }
LABEL_71:
            v105 = (char *)src + 8 * (unsigned int)v114;
            if ( src == v105 )
              break;
            v107 = (__int64 *)src;
            while ( 1 )
            {
              v63 = *(__int64 ***)(v6 + 752);
              v64 = *v107;
              v65 = 8LL * *(unsigned int *)(v6 + 760);
              v111 = &v63[(unsigned __int64)v65 / 8];
              v66 = v65 >> 3;
              v67 = v65 >> 5;
              if ( !v67 )
                goto LABEL_111;
              v68 = &v63[4 * v67];
              do
              {
                if ( (unsigned __int8)sub_E38870(*v63, (__int64 *)v64) )
                  goto LABEL_80;
                if ( (unsigned __int8)sub_E38870(v63[1], (__int64 *)v64) )
                {
                  ++v63;
LABEL_80:
                  if ( v111 == v63 )
                    goto LABEL_115;
                  goto LABEL_81;
                }
                if ( (unsigned __int8)sub_E38870(v63[2], (__int64 *)v64) )
                {
                  v63 += 2;
                  goto LABEL_80;
                }
                if ( (unsigned __int8)sub_E38870(v63[3], (__int64 *)v64) )
                {
                  v63 += 3;
                  goto LABEL_80;
                }
                v63 += 4;
              }
              while ( v63 != v68 );
              v66 = v111 - v63;
LABEL_111:
              if ( v66 == 2 )
                goto LABEL_137;
              if ( v66 != 3 )
              {
                if ( v66 == 1 )
                  goto LABEL_114;
                goto LABEL_115;
              }
              if ( (unsigned __int8)sub_E38870(*v63, (__int64 *)v64) )
                goto LABEL_80;
              ++v63;
LABEL_137:
              if ( (unsigned __int8)sub_E38870(*v63, (__int64 *)v64) )
                goto LABEL_80;
              ++v63;
LABEL_114:
              if ( (unsigned __int8)sub_E38870(*v63, (__int64 *)v64) )
                goto LABEL_80;
LABEL_115:
              v88 = *(unsigned int *)(v6 + 760);
              v89 = *(unsigned int *)(v6 + 764);
              if ( v88 + 1 > v89 )
              {
                sub_C8D5F0(v6 + 752, (const void *)(v6 + 768), v88 + 1, 8u, v61, v62);
                v88 = *(unsigned int *)(v6 + 760);
              }
              *(_QWORD *)(*(_QWORD *)(v6 + 752) + 8 * v88) = v64;
              ++*(_DWORD *)(v6 + 760);
              v90 = *(_QWORD *)(v64 + 88);
              v109 = v90 + 8LL * *(unsigned int *)(v64 + 96);
              if ( v90 != v109 )
              {
                while ( 1 )
                {
                  v91 = *(_QWORD *)(*(_QWORD *)v90 + 56LL);
                  v92 = *(_QWORD *)v90 + 48LL;
                  if ( v92 != v91 )
                    break;
LABEL_127:
                  v90 += 8;
                  if ( v109 == v90 )
                    goto LABEL_81;
                }
                while ( 1 )
                {
                  if ( !v91 )
                    BUG();
                  v93 = (unsigned __int8 *)(v91 - 24);
                  if ( (unsigned int)*(unsigned __int8 *)(v91 - 24) - 30 <= 0xA )
                    goto LABEL_127;
                  if ( *(_BYTE *)(v6 + 1284) )
                  {
                    v94 = *(__int64 **)(v6 + 1264);
                    v95 = &v94[*(unsigned int *)(v6 + 1276)];
                    if ( v94 != v95 )
                    {
                      while ( v93 != (unsigned __int8 *)*v94 )
                      {
                        if ( v95 == ++v94 )
                          goto LABEL_130;
                      }
LABEL_126:
                      v91 = *(_QWORD *)(v91 + 8);
                      if ( v92 == v91 )
                        goto LABEL_127;
                      continue;
                    }
                  }
                  else
                  {
                    v96 = sub_C8CA60(v6 + 1256, (__int64)v93);
                    v93 = (unsigned __int8 *)(v91 - 24);
                    if ( v96 )
                      goto LABEL_126;
                  }
LABEL_130:
                  sub_1057F60(v6, v93, v95, v89, v61, v62);
                  v91 = *(_QWORD *)(v91 + 8);
                  if ( v92 == v91 )
                    goto LABEL_127;
                }
              }
LABEL_81:
              if ( v105 == (char *)++v107 )
                goto LABEL_15;
            }
          }
        }
      }
LABEL_15:
      v16 = v102;
      v17 = sub_E387E0(*(_QWORD *)(v6 + 224), v102);
      result = (__int64 *)v101[9];
      if ( *((_BYTE *)v101 + 92) )
        v18 = *((unsigned int *)v101 + 21);
      else
        v18 = *((unsigned int *)v101 + 20);
      v19 = &result[v18];
      if ( result != v19 )
      {
        while ( 1 )
        {
          v16 = *result;
          v20 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v19 == ++result )
            goto LABEL_20;
        }
        while ( v20 != v19 )
        {
          sub_10587E0(v6, v16, v17);
          result = v20 + 1;
          if ( v20 + 1 == v19 )
            break;
          while ( 1 )
          {
            v16 = *result;
            v20 = result;
            if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v19 == ++result )
              goto LABEL_20;
          }
        }
      }
LABEL_20:
      if ( src != v115 )
        return (__int64 *)_libc_free(src, v16);
    }
  }
  return result;
}
