// Function: sub_2B87C80
// Address: 0x2b87c80
//
__int64 __fastcall sub_2B87C80(__int64 a1, char ***a2)
{
  __int64 v2; // r9
  unsigned int v4; // r8d
  __int64 v5; // rcx
  char ***v6; // r10
  int v7; // r13d
  char ****v8; // r11
  unsigned int v9; // eax
  char ****v10; // r12
  char ***v11; // rdx
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  char ***v15; // rdi
  char **v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 v21; // rcx
  unsigned int v22; // r8d
  char **v23; // rbx
  __int64 v24; // rdx
  unsigned __int8 **v25; // r14
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned __int8 **v28; // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // ecx
  __int64 *v32; // rdx
  __int64 v33; // r9
  __int64 v34; // r14
  unsigned __int8 v35; // dl
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  _QWORD *v40; // rax
  _QWORD *v41; // rcx
  int v42; // eax
  __int64 v43; // rsi
  int v44; // edi
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r8
  __int64 v48; // rdx
  char **v49; // r15
  __int64 v50; // r9
  __int64 v51; // r9
  __int64 v52; // r9
  __int64 *v53; // r14
  __int64 v54; // rbx
  __int64 *v55; // r13
  __int64 v56; // rbx
  __int64 *v57; // rbx
  unsigned __int8 *v58; // rdi
  int v59; // eax
  unsigned __int8 *v60; // rdi
  int v61; // eax
  unsigned __int8 *v62; // rdi
  int v63; // eax
  int v64; // eax
  __int64 v65; // r9
  char **v66; // r15
  __int64 v67; // r9
  int v68; // eax
  char ***v69; // rsi
  int v70; // ecx
  __int64 v71; // rdi
  unsigned int v72; // eax
  int v73; // r9d
  char ****v74; // r8
  int v75; // eax
  int v76; // ecx
  __int64 v77; // rdi
  int v78; // r9d
  unsigned int v79; // eax
  int v80; // edx
  int v81; // r10d
  __int64 v82; // r9
  __int64 v83; // r9
  int v84; // eax
  int v85; // r9d
  signed __int64 v86; // rax
  int v87; // eax
  int v88; // eax
  int v89; // eax
  char ***v90; // [rsp+8h] [rbp-88h] BYREF
  char **v91; // [rsp+10h] [rbp-80h] BYREF
  __int64 v92; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v93[4]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v94[10]; // [rsp+40h] [rbp-50h] BYREF

  v2 = a1 + 1088;
  v4 = *(_DWORD *)(a1 + 1112);
  v90 = a2;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 1088);
    goto LABEL_122;
  }
  v5 = *(_QWORD *)(a1 + 1096);
  v6 = a2;
  v7 = 1;
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (char ****)(v5 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    result = (__int64)v10[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != (char ***)-4096LL )
  {
    if ( v11 == (char ***)-8192LL && !v8 )
      v8 = v10;
    v9 = (v4 - 1) & (v7 + v9);
    v10 = (char ****)(v5 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  v13 = *(_DWORD *)(a1 + 1104);
  if ( v8 )
    v10 = v8;
  ++*(_QWORD *)(a1 + 1088);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_122:
    sub_2B593D0(v2, 2 * v4);
    v68 = *(_DWORD *)(a1 + 1112);
    if ( v68 )
    {
      v69 = v90;
      v70 = v68 - 1;
      v71 = *(_QWORD *)(a1 + 1096);
      v72 = (v68 - 1) & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
      v14 = *(_DWORD *)(a1 + 1104) + 1;
      v10 = (char ****)(v71 + 16LL * v72);
      v6 = *v10;
      if ( v90 == *v10 )
        goto LABEL_15;
      v73 = 1;
      v74 = 0;
      while ( v6 != (char ***)-4096LL )
      {
        if ( !v74 && v6 == (char ***)-8192LL )
          v74 = v10;
        v72 = v70 & (v73 + v72);
        v10 = (char ****)(v71 + 16LL * v72);
        v6 = *v10;
        if ( v90 == *v10 )
          goto LABEL_15;
        ++v73;
      }
LABEL_126:
      v6 = v69;
      if ( v74 )
        v10 = v74;
      goto LABEL_15;
    }
LABEL_187:
    ++*(_DWORD *)(a1 + 1104);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 1108) - v14 <= v4 >> 3 )
  {
    sub_2B593D0(v2, v4);
    v75 = *(_DWORD *)(a1 + 1112);
    if ( v75 )
    {
      v69 = v90;
      v76 = v75 - 1;
      v77 = *(_QWORD *)(a1 + 1096);
      v74 = 0;
      v78 = 1;
      v79 = (v75 - 1) & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
      v14 = *(_DWORD *)(a1 + 1104) + 1;
      v10 = (char ****)(v77 + 16LL * v79);
      v6 = *v10;
      if ( v90 == *v10 )
        goto LABEL_15;
      while ( v6 != (char ***)-4096LL )
      {
        if ( !v74 && v6 == (char ***)-8192LL )
          v74 = v10;
        v79 = v76 & (v78 + v79);
        v10 = (char ****)(v77 + 16LL * v79);
        v6 = *v10;
        if ( v90 == *v10 )
          goto LABEL_15;
        ++v78;
      }
      goto LABEL_126;
    }
    goto LABEL_187;
  }
LABEL_15:
  *(_DWORD *)(a1 + 1104) = v14;
  if ( *v10 != (char ***)-4096LL )
    --*(_DWORD *)(a1 + 1108);
  v10[1] = 0;
  *v10 = v6;
  a2 = v90;
LABEL_18:
  v94[2] = a1;
  v15 = a2;
  v16 = a2[52];
  v94[0] = &v91;
  v94[3] = &v92;
  v17 = (__int64)v16[5];
  v91 = v16;
  v93[0] = &v91;
  v92 = v17;
  v94[1] = &v90;
  v93[1] = &v90;
  v18 = *((_DWORD *)a2 + 26);
  v93[2] = a1;
  if ( v18 == 5 )
    goto LABEL_35;
  if ( *(_BYTE *)(a1 + 1256) && *((_DWORD *)a2 + 50) >= *(_DWORD *)(a1 + 1252) && v18 != 3 && *(_BYTE *)v16 == 61 )
    goto LABEL_86;
  v19 = *((unsigned int *)a2 + 2);
  if ( *((_DWORD *)a2 + 2) )
  {
    v66 = *a2;
    v20 = *((unsigned int *)a2 + 2);
    if ( sub_2B0D880(v66, v20, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B099C0)
      || (v20 = v19, sub_2B0D880(v66, v19, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B16010)) )
    {
LABEL_26:
      v23 = *v90;
      v24 = 8LL * *((unsigned int *)v90 + 2);
      v25 = (unsigned __int8 **)&(*v90)[(unsigned __int64)v24 / 8];
      v26 = v24 >> 5;
      v27 = v24 >> 3;
      if ( *(_BYTE *)v90[52] != 63 )
        goto LABEL_102;
      if ( v26 )
      {
        v28 = (unsigned __int8 **)*v90;
        v20 = v26;
        while ( 1 )
        {
          v21 = **v28;
          if ( (_BYTE)v21 != 63 && (unsigned __int8)v21 > 0x1Cu )
            break;
          v21 = *v28[1];
          if ( (unsigned __int8)v21 > 0x1Cu && (_BYTE)v21 != 63 )
          {
            ++v28;
            break;
          }
          v21 = *v28[2];
          if ( (_BYTE)v21 != 63 && (unsigned __int8)v21 > 0x1Cu )
          {
            v28 += 2;
            break;
          }
          v21 = *v28[3];
          if ( (_BYTE)v21 != 63 && (unsigned __int8)v21 > 0x1Cu )
          {
            v28 += 3;
            break;
          }
          v28 += 4;
          if ( !--v20 )
          {
            v21 = v25 - v28;
            goto LABEL_96;
          }
        }
        if ( v28 != v25 )
          goto LABEL_35;
LABEL_68:
        while ( **v23 == 13
             || !(unsigned __int8)sub_2B15E10(*v23, v20, v27, v21, v22) && (unsigned __int8)sub_2B099C0(v52) )
        {
          v49 = v23 + 1;
          if ( *v23[1] != 13
            && ((unsigned __int8)sub_2B15E10(v23[1], v20, v27, v21, v22) || !(unsigned __int8)sub_2B099C0(v65))
            || (v49 = v23 + 2, *v23[2] != 13)
            && ((unsigned __int8)sub_2B15E10(v23[2], v20, v27, v21, v22) || !(unsigned __int8)sub_2B099C0(v50))
            || (v49 = v23 + 3, *v23[3] != 13)
            && ((unsigned __int8)sub_2B15E10(v23[3], v20, v27, v21, v22) || !(unsigned __int8)sub_2B099C0(v51)) )
          {
            v23 = v49;
            goto LABEL_70;
          }
          v23 += 4;
          if ( !--v26 )
          {
            v27 = ((char *)v25 - (char *)v23) >> 3;
            goto LABEL_105;
          }
        }
        goto LABEL_70;
      }
      v21 = v27;
      v28 = (unsigned __int8 **)*v90;
LABEL_96:
      if ( v21 != 2 )
      {
        if ( v21 != 3 )
        {
          if ( v21 != 1 )
            goto LABEL_102;
LABEL_99:
          v21 = **v28;
          if ( (_BYTE)v21 == 63 || (unsigned __int8)v21 <= 0x1Cu )
            goto LABEL_102;
          goto LABEL_101;
        }
        v21 = **v28;
        if ( (unsigned __int8)v21 > 0x1Cu && (_BYTE)v21 != 63 )
        {
LABEL_101:
          if ( v28 != v25 )
            goto LABEL_35;
LABEL_102:
          if ( v26 )
            goto LABEL_68;
LABEL_105:
          if ( v27 != 2 )
          {
            if ( v27 != 3 )
            {
              if ( v27 != 1 )
                goto LABEL_35;
LABEL_108:
              if ( **v23 == 13
                || !(unsigned __int8)sub_2B15E10(*v23, v20, v27, v21, v22) && (unsigned __int8)sub_2B099C0(v67) )
              {
                goto LABEL_35;
              }
LABEL_70:
              if ( v23 == (char **)v25 )
                goto LABEL_35;
              if ( *((_DWORD *)v90 + 26) != 3 || *((_DWORD *)v90 + 50) )
                goto LABEL_86;
              v53 = (__int64 *)*v90;
              v54 = *((unsigned int *)v90 + 2);
              v55 = (__int64 *)&(*v90)[v54];
              v56 = (v54 * 8) >> 5;
              if ( v56 )
              {
                v57 = &v53[4 * v56];
                while ( 1 )
                {
                  v64 = *(unsigned __int8 *)*v53;
                  if ( (_BYTE)v64 != 90 && (unsigned int)(v64 - 12) > 1 && !sub_2B16010(*v53) )
                    break;
                  v58 = (unsigned __int8 *)v53[1];
                  v59 = *v58;
                  if ( (_BYTE)v59 != 90 && (unsigned int)(v59 - 12) > 1 && !sub_2B16010((__int64)v58) )
                  {
                    ++v53;
                    break;
                  }
                  v60 = (unsigned __int8 *)v53[2];
                  v61 = *v60;
                  if ( (_BYTE)v61 != 90 && (unsigned int)(v61 - 12) > 1 && !sub_2B16010((__int64)v60) )
                  {
                    v53 += 2;
                    break;
                  }
                  v62 = (unsigned __int8 *)v53[3];
                  v63 = *v62;
                  if ( (_BYTE)v63 != 90 && (unsigned int)(v63 - 12) > 1 && !sub_2B16010((__int64)v62) )
                  {
                    v53 += 3;
                    break;
                  }
                  v53 += 4;
                  if ( v53 == v57 )
                    goto LABEL_158;
                }
LABEL_85:
                if ( v55 != v53 )
                {
LABEL_86:
                  result = sub_2B10CE0((__int64)v93);
                  v10[1] = (char ***)result;
                  return result;
                }
LABEL_35:
                result = sub_2B10DF0((__int64)v94);
                v10[1] = (char ***)result;
                return result;
              }
              v57 = (__int64 *)*v90;
LABEL_158:
              v86 = (char *)v55 - (char *)v57;
              if ( (char *)v55 - (char *)v57 != 16 )
              {
                if ( v86 != 24 )
                {
                  if ( v86 != 8 )
                    goto LABEL_35;
                  goto LABEL_161;
                }
                v88 = *(unsigned __int8 *)*v57;
                if ( (_BYTE)v88 != 90 && (unsigned int)(v88 - 12) > 1 && !sub_2B16010(*v57) )
                {
LABEL_164:
                  v53 = v57;
                  goto LABEL_85;
                }
                ++v57;
              }
              v89 = *(unsigned __int8 *)*v57;
              if ( (_BYTE)v89 == 90 || (unsigned int)(v89 - 12) <= 1 || sub_2B16010(*v57) )
              {
                ++v57;
LABEL_161:
                v87 = *(unsigned __int8 *)*v57;
                if ( (_BYTE)v87 == 90 || (unsigned int)(v87 - 12) <= 1 || sub_2B16010(*v57) )
                  goto LABEL_35;
                goto LABEL_164;
              }
              goto LABEL_164;
            }
            if ( **v23 != 13
              && ((unsigned __int8)sub_2B15E10(*v23, v20, 3, v21, v22) || !(unsigned __int8)sub_2B099C0(v82)) )
            {
              goto LABEL_70;
            }
            ++v23;
          }
          if ( **v23 != 13
            && ((unsigned __int8)sub_2B15E10(*v23, v20, v27, v21, v22) || !(unsigned __int8)sub_2B099C0(v83)) )
          {
            goto LABEL_70;
          }
          ++v23;
          goto LABEL_108;
        }
        ++v28;
      }
      v21 = **v28;
      if ( (_BYTE)v21 == 63 || (unsigned __int8)v21 <= 0x1Cu )
      {
        ++v28;
        goto LABEL_99;
      }
      goto LABEL_101;
    }
    v15 = v90;
    v18 = *((_DWORD *)v90 + 26);
  }
  if ( v18 != 3 )
  {
    v20 = (__int64)&(*v15)[*((unsigned int *)v15 + 2)];
    if ( (_QWORD *)v20 == sub_2B0BF30(*v15, v20, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B15E10) )
      goto LABEL_26;
  }
  v29 = *(unsigned int *)(a1 + 3248);
  v30 = *(_QWORD *)(a1 + 3232);
  if ( !(_DWORD)v29 )
    goto LABEL_57;
  v31 = (v29 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
  v32 = (__int64 *)(v30 + 16LL * v31);
  v33 = *v32;
  if ( v92 != *v32 )
  {
    v80 = 1;
    while ( v33 != -4096 )
    {
      v81 = v80 + 1;
      v31 = (v29 - 1) & (v80 + v31);
      v32 = (__int64 *)(v30 + 16LL * v31);
      v33 = *v32;
      if ( v92 == *v32 )
        goto LABEL_38;
      v80 = v81;
    }
    goto LABEL_57;
  }
LABEL_38:
  if ( v32 == (__int64 *)(v30 + 16 * v29) || *((_DWORD *)v90 + 26) == 3 )
    goto LABEL_57;
  v34 = (__int64)v90[52];
  v35 = *(*v90)[*((unsigned int *)v90 + 2) - 1];
  if ( v35 > 0x1Cu )
  {
    if ( v35 == *(_BYTE *)v34 )
    {
      v34 = (__int64)(*v90)[*((unsigned int *)v90 + 2) - 1];
    }
    else if ( v35 == *(_BYTE *)v90[53] )
    {
      v34 = (__int64)(*v90)[*((unsigned int *)v90 + 2) - 1];
    }
  }
  if ( (unsigned __int8)sub_2B14730(v34) )
    v34 = *sub_2B0BF30(
             *v90,
             (__int64)&(*v90)[*((unsigned int *)v90 + 2)],
             (unsigned __int8 (__fastcall *)(_QWORD))sub_2B14730);
  v40 = (_QWORD *)sub_2B87810(a1 + 3224, &v92, v36, v37, v38, v39);
  if ( *(_BYTE *)v34 <= 0x1Cu )
    goto LABEL_57;
  v41 = (_QWORD *)*v40;
  if ( *(_QWORD *)*v40 != *(_QWORD *)(v34 + 40) )
    goto LABEL_57;
  v42 = *((_DWORD *)v41 + 26);
  v43 = v41[11];
  if ( !v42 )
    goto LABEL_57;
  v44 = v42 - 1;
  v45 = (v42 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
  v46 = (__int64 *)(v43 + 16LL * v45);
  v47 = *v46;
  if ( v34 != *v46 )
  {
    v84 = 1;
    while ( v47 != -4096 )
    {
      v85 = v84 + 1;
      v45 = v44 & (v84 + v45);
      v46 = (__int64 *)(v43 + 16LL * v45);
      v47 = *v46;
      if ( v34 == *v46 )
        goto LABEL_50;
      v84 = v85;
    }
    goto LABEL_57;
  }
LABEL_50:
  v48 = v46[1];
  if ( !v48
    || *(_DWORD *)(v48 + 136) != *((_DWORD *)v41 + 51)
    || !*(_QWORD *)(v48 + 24) && v48 == *(_QWORD *)(v48 + 16) && !*(_QWORD *)(v48 + 8) )
  {
LABEL_57:
    result = (__int64)v10[1];
    goto LABEL_58;
  }
  do
  {
    result = *(_QWORD *)v48;
    v10[1] = *(char ****)v48;
    v48 = *(_QWORD *)(v48 + 24);
  }
  while ( v48 );
LABEL_58:
  if ( !result )
    goto LABEL_35;
  return result;
}
