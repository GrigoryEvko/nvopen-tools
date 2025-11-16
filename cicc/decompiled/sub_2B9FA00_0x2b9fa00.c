// Function: sub_2B9FA00
// Address: 0x2b9fa00
//
char **__fastcall sub_2B9FA00(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r9
  __int64 *v7; // r13
  unsigned __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 *v17; // r15
  __int64 *v18; // rax
  __int64 *v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rdx
  _QWORD *v29; // rcx
  __int64 v30; // rsi
  int v31; // edx
  int v32; // edx
  int v33; // edx
  int v34; // edx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rax
  int v41; // r10d
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r12
  _QWORD *v45; // rdi
  _QWORD *v46; // r8
  __int64 v47; // r9
  unsigned int v48; // r10d
  __int64 v49; // r11
  __int64 v50; // rbx
  int v51; // edx
  __int64 v52; // rax
  _DWORD *v53; // r14
  _DWORD *v54; // rax
  unsigned __int64 v55; // rcx
  char *v56; // rdx
  __int64 v57; // rsi
  _QWORD *v58; // rax
  __int64 v59; // rsi
  __int64 *v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rsi
  _QWORD *v63; // rdx
  __int64 v64; // rbx
  char *v65; // rax
  int v66; // ecx
  unsigned __int64 v67; // rcx
  unsigned __int64 v68; // r8
  __int64 v69; // rax
  unsigned __int64 v70; // rsi
  __int64 v71; // rdx
  __int64 *v72; // r8
  __int64 *v73; // rdi
  int v74; // edx
  int v75; // edx
  int v76; // eax
  int v77; // esi
  unsigned int *v78; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v79; // [rsp+18h] [rbp-D8h]
  __int64 v80; // [rsp+20h] [rbp-D0h]
  __int64 v83; // [rsp+38h] [rbp-B8h]
  char *v84; // [rsp+38h] [rbp-B8h]
  int v85; // [rsp+40h] [rbp-B0h]
  __int64 v86; // [rsp+40h] [rbp-B0h]
  __int64 v87; // [rsp+48h] [rbp-A8h] BYREF
  int v88; // [rsp+54h] [rbp-9Ch] BYREF
  __int64 v89; // [rsp+58h] [rbp-98h] BYREF
  _QWORD v90[4]; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v91; // [rsp+80h] [rbp-70h] BYREF
  __int64 v92; // [rsp+88h] [rbp-68h]
  _BYTE v93[96]; // [rsp+90h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a2 + 240) + 80LL * a3;
  v87 = a2;
  v4 = sub_2B5F980(*(__int64 **)v3, *(unsigned int *)(v3 + 8), *(__int64 **)(a1 + 3304));
  v7 = *(__int64 **)v3;
  if ( v4 && v5 )
  {
    v8 = *(unsigned int *)(v3 + 8);
    v85 = *(_DWORD *)(v3 + 8);
    v83 = v87;
    goto LABEL_4;
  }
  v8 = *(unsigned int *)(v3 + 8);
  v85 = *(_DWORD *)(v3 + 8);
  if ( *(_BYTE *)(*(_QWORD *)(*v7 + 8) + 8LL) == 14 )
  {
    v72 = &v7[v8];
    v6 = (__int64)(8 * v8) >> 3;
    if ( (__int64)(8 * v8) >> 5 )
    {
      v73 = *(__int64 **)v3;
      if ( *(_BYTE *)*v7 == 63 )
        goto LABEL_128;
      while ( 1 )
      {
        if ( *(_BYTE *)v73[1] == 63 )
        {
          ++v73;
          goto LABEL_128;
        }
        if ( *(_BYTE *)v73[2] == 63 )
        {
          v73 += 2;
          goto LABEL_128;
        }
        if ( *(_BYTE *)v73[3] == 63 )
        {
          v73 += 3;
          goto LABEL_128;
        }
        v73 += 4;
        if ( &v7[4 * ((__int64)(8 * v8) >> 5)] == v73 )
          break;
        if ( *(_BYTE *)*v73 == 63 )
          goto LABEL_128;
      }
      v6 = v72 - v73;
    }
    else
    {
      v73 = *(__int64 **)v3;
    }
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_38;
        goto LABEL_190;
      }
      if ( *(_BYTE *)*v73 == 63 )
      {
LABEL_128:
        if ( v72 != v73 )
        {
          v4 = sub_2B5F980(v73, 1u, *(__int64 **)(a1 + 3304));
          v8 = *(unsigned int *)(v3 + 8);
          v7 = *(__int64 **)v3;
          v85 = *(_DWORD *)(v3 + 8);
        }
        goto LABEL_38;
      }
      ++v73;
    }
    if ( *(_BYTE *)*v73 != 63 )
    {
      ++v73;
LABEL_190:
      if ( *(_BYTE *)*v73 != 63 )
        goto LABEL_38;
      goto LABEL_128;
    }
    goto LABEL_128;
  }
LABEL_38:
  v83 = v87;
  v22 = v87;
  if ( !v4 || !v5 )
    goto LABEL_18;
LABEL_4:
  if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
  {
    v9 = a1 + 96;
    v10 = 3;
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 96);
    v66 = *(_DWORD *)(a1 + 104);
    if ( !v66 )
      goto LABEL_115;
    v10 = (unsigned int)(v66 - 1);
  }
  v11 = (unsigned int)v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v12 = v9 + 72 * v11;
  v13 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 != v4 )
  {
    v77 = 1;
    while ( v13 != -4096 )
    {
      v6 = (unsigned int)(v77 + 1);
      v11 = (unsigned int)v10 & (v77 + (_DWORD)v11);
      v12 = v9 + 72LL * (unsigned int)v11;
      v13 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == v4 )
        goto LABEL_7;
      v77 = v6;
    }
    goto LABEL_115;
  }
LABEL_7:
  v91 = (__int64 *)v93;
  v92 = 0x600000000LL;
  if ( !*(_DWORD *)(v12 + 16) )
  {
LABEL_115:
    v22 = v83;
    goto LABEL_18;
  }
  sub_2B0C870((__int64)&v91, v12 + 8, v11, v10, v13, v6);
  v16 = (__int64)v91;
  v17 = &v91[(unsigned int)v92];
  v18 = v17;
  if ( v91 == v17 )
  {
LABEL_112:
    if ( v18 != (__int64 *)v93 )
      _libc_free((unsigned __int64)v18);
LABEL_114:
    v83 = v87;
    goto LABEL_115;
  }
  v78 = (unsigned int *)v3;
  v19 = v91;
  while ( 1 )
  {
    v20 = *v19;
    if ( sub_2B31C30(*v19, (char *)v7, v8, v16, v14, v15) )
      break;
    if ( v17 == ++v19 )
    {
      v18 = v91;
      goto LABEL_112;
    }
  }
  v21 = v20;
  if ( v91 != (__int64 *)v93 )
    _libc_free((unsigned __int64)v91);
  if ( !v20 )
    goto LABEL_114;
  if ( *(_QWORD *)(v20 + 184) == v83 && *(_DWORD *)(v20 + 192) == a3 )
  {
    v90[0] = v78;
    v90[2] = &v87;
    v90[1] = a1;
    v84 = (char *)sub_2B9A8C0(a1, (char ***)v20);
    v38 = *(_QWORD *)(**(_QWORD **)v78 + 8LL);
    if ( *(_BYTE *)(v38 + 8) == 17 )
      v85 *= *(_DWORD *)(v38 + 32);
    if ( v85 != *(_DWORD *)(*((_QWORD *)v84 + 1) + 32LL) )
    {
      v39 = *(unsigned int *)(v20 + 120);
      v91 = (__int64 *)v93;
      v92 = 0xC00000000LL;
      if ( (_DWORD)v39 )
      {
        sub_11B1960((__int64)&v91, v8, -1, v39, v36, v37);
        v40 = *(_QWORD *)v78 + 8LL * v78[2];
        v80 = *(_QWORD *)v78;
        if ( v40 != *(_QWORD *)v78 )
        {
          v86 = 0;
          v79 = 4 * ((unsigned __int64)(v40 - v80 - 8) >> 3) + 4;
          do
          {
            if ( **(_BYTE **)(v80 + 2 * v86) != 13 )
            {
              v41 = *(_DWORD *)(v21 + 120);
              v42 = *(unsigned int *)(v21 + 8);
              v89 = *(_QWORD *)(v80 + 2 * v86);
              LODWORD(v43) = v41;
              if ( !v41 )
                LODWORD(v43) = v42;
              v44 = *(_QWORD *)v21 + 8 * v42;
              v45 = *(_QWORD **)v21;
              v88 = v43;
              v46 = sub_2B0CA10(v45, v44, &v89);
              if ( (_QWORD *)v44 != v46 )
              {
                v50 = v48;
                while ( 1 )
                {
                  if ( v47 == *v46 )
                  {
                    v51 = *(_DWORD *)(v21 + 152);
                    v52 = ((__int64)v46 - v49) >> 3;
                    v88 = v52;
                    if ( v51 )
                      v88 = *(_DWORD *)(*(_QWORD *)(v21 + 144) + 4LL * (unsigned int)v52);
                    if ( !v48 )
                    {
LABEL_130:
                      LODWORD(v43) = v88;
                      break;
                    }
                    v53 = *(_DWORD **)(v21 + 112);
                    v54 = sub_2B0C950(v53, (__int64)&v53[v50], &v88);
                    if ( &v53[v50] != v54 )
                    {
                      v43 = v54 - v53;
                      break;
                    }
                  }
                  if ( (_QWORD *)v44 == ++v46 )
                    goto LABEL_130;
                }
              }
              *(_DWORD *)((char *)v91 + v86) = v43;
            }
            v86 += 4;
          }
          while ( v79 != v86 );
        }
        v55 = (unsigned int)v92;
        v56 = (char *)v91;
      }
      else
      {
        sub_11B1960((__int64)&v91, v8, 0, v39, v36, v37);
        v67 = (unsigned __int64)v91;
        v68 = (unsigned int)v92;
        v56 = (char *)v91 + 4 * (unsigned int)v92;
        if ( v91 != (__int64 *)v56 )
        {
          v69 = 0;
          v70 = (4 * (unsigned __int64)(unsigned int)v92 - 4) >> 2;
          do
          {
            v71 = v69;
            *(_DWORD *)(v67 + 4 * v69) = v69;
            ++v69;
          }
          while ( v70 != v71 );
          v56 = (char *)v91;
          v68 = (unsigned int)v92;
        }
        v55 = v68;
      }
      v84 = (char *)sub_2B7D590(v90, (__int64)v84, v56, v55);
      if ( v91 != (__int64 *)v93 )
        _libc_free((unsigned __int64)v91);
    }
    if ( *(_QWORD *)(v21 + 184) == v87 && *(_DWORD *)(v21 + 192) == a3 )
      return (char **)v84;
    v57 = (unsigned int)(*(_DWORD *)(v87 + 200) + 1);
    v58 = (_QWORD *)(*(_QWORD *)a1 + 8 * v57);
    v59 = 8 * (*(unsigned int *)(a1 + 8) - v57);
    v60 = &v58[(unsigned __int64)v59 / 8];
    v61 = v59 >> 5;
    v62 = v59 >> 3;
    if ( v61 > 0 )
    {
      v63 = &v58[4 * v61];
      while ( 1 )
      {
        v64 = *v58;
        if ( *(_DWORD *)(*v58 + 104LL) == 3 && v87 == *(_QWORD *)(v64 + 184) && *(_DWORD *)(v64 + 192) == a3 )
          goto LABEL_74;
        v64 = v58[1];
        if ( *(_DWORD *)(v64 + 104) == 3 && v87 == *(_QWORD *)(v64 + 184) )
        {
          if ( *(_DWORD *)(v64 + 192) == a3 )
            goto LABEL_74;
          v64 = v58[2];
          if ( *(_DWORD *)(v64 + 104) != 3 )
          {
LABEL_69:
            v64 = v58[3];
            if ( *(_DWORD *)(v64 + 104) == 3 )
              goto LABEL_96;
            goto LABEL_70;
          }
        }
        else
        {
          v64 = v58[2];
          if ( *(_DWORD *)(v64 + 104) != 3 )
            goto LABEL_69;
        }
        if ( v87 != *(_QWORD *)(v64 + 184) )
          goto LABEL_69;
        if ( *(_DWORD *)(v64 + 192) == a3 )
          goto LABEL_74;
        v64 = v58[3];
        if ( *(_DWORD *)(v64 + 104) == 3 )
        {
LABEL_96:
          if ( v87 == *(_QWORD *)(v64 + 184) && *(_DWORD *)(v64 + 192) == a3 )
            goto LABEL_74;
        }
LABEL_70:
        v58 += 4;
        if ( v63 == v58 )
        {
          v62 = v60 - v58;
          break;
        }
      }
    }
    if ( v62 != 2 )
    {
      if ( v62 != 3 )
      {
        if ( v62 != 1 )
        {
LABEL_135:
          v64 = *v60;
          goto LABEL_74;
        }
LABEL_145:
        v64 = *v58;
        if ( *(_DWORD *)(*v58 + 104LL) != 3 || v87 != *(_QWORD *)(v64 + 184) || *(_DWORD *)(v64 + 192) != a3 )
          goto LABEL_135;
LABEL_74:
        v65 = *(char **)(v64 + 96);
        if ( v84 != v65 )
        {
          if ( v65 != 0 && v65 + 4096 != 0 && v65 != (char *)-8192LL )
            sub_BD60C0((_QWORD *)(v64 + 80));
          *(_QWORD *)(v64 + 96) = v84;
          if ( v84 != 0 && v84 + 4096 != 0 && v84 != (char *)-8192LL )
            sub_BD73F0(v64 + 80);
        }
        return (char **)v84;
      }
      v64 = *v58;
      if ( *(_DWORD *)(*v58 + 104LL) == 3 && v87 == *(_QWORD *)(v64 + 184) && *(_DWORD *)(v64 + 192) == a3 )
        goto LABEL_74;
      ++v58;
    }
    v64 = *v58;
    if ( *(_DWORD *)(*v58 + 104LL) == 3 && v87 == *(_QWORD *)(v64 + 184) && *(_DWORD *)(v64 + 192) == a3 )
      goto LABEL_74;
    ++v58;
    goto LABEL_145;
  }
  v22 = v87;
LABEL_18:
  v23 = (unsigned int)(*(_DWORD *)(v22 + 200) + 1);
  v24 = (_QWORD *)(*(_QWORD *)a1 + 8 * v23);
  v25 = 8 * (*(unsigned int *)(a1 + 8) - v23);
  v26 = &v24[(unsigned __int64)v25 / 8];
  v27 = v25 >> 5;
  v28 = v25 >> 3;
  if ( v27 <= 0 )
    goto LABEL_137;
  v29 = &v24[4 * v27];
  do
  {
    v30 = *v24;
    v34 = *(_DWORD *)(*v24 + 104LL);
    if ( v34 == 3 )
    {
      if ( a3 == *(_DWORD *)(v30 + 192) && *(_QWORD *)(v30 + 184) == v22 )
        return sub_2B9A8C0(a1, (char ***)v30);
    }
    else if ( v34 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
    {
      return sub_2B9A8C0(a1, (char ***)v30);
    }
    v30 = v24[1];
    v31 = *(_DWORD *)(v30 + 104);
    if ( v31 != 3 )
    {
      if ( v31 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
        return sub_2B9A8C0(a1, (char ***)v30);
LABEL_25:
      v30 = v24[2];
      v32 = *(_DWORD *)(v30 + 104);
      if ( v32 != 3 )
        goto LABEL_26;
      goto LABEL_84;
    }
    if ( a3 != *(_DWORD *)(v30 + 192) )
      goto LABEL_25;
    if ( *(_QWORD *)(v30 + 184) == v22 )
      return sub_2B9A8C0(a1, (char ***)v30);
    v30 = v24[2];
    v32 = *(_DWORD *)(v30 + 104);
    if ( v32 != 3 )
    {
LABEL_26:
      if ( v32 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
        return sub_2B9A8C0(a1, (char ***)v30);
LABEL_28:
      v30 = v24[3];
      v33 = *(_DWORD *)(v30 + 104);
      if ( v33 != 3 )
        goto LABEL_29;
      goto LABEL_87;
    }
LABEL_84:
    if ( a3 != *(_DWORD *)(v30 + 192) )
      goto LABEL_28;
    if ( *(_QWORD *)(v30 + 184) == v22 )
      return sub_2B9A8C0(a1, (char ***)v30);
    v30 = v24[3];
    v33 = *(_DWORD *)(v30 + 104);
    if ( v33 != 3 )
    {
LABEL_29:
      if ( v33 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
        return sub_2B9A8C0(a1, (char ***)v30);
      goto LABEL_31;
    }
LABEL_87:
    if ( a3 == *(_DWORD *)(v30 + 192) && *(_QWORD *)(v30 + 184) == v22 )
      return sub_2B9A8C0(a1, (char ***)v30);
LABEL_31:
    v24 += 4;
  }
  while ( v24 != v29 );
  v28 = v26 - v24;
LABEL_137:
  if ( v28 == 2 )
    goto LABEL_153;
  if ( v28 == 3 )
  {
    v30 = *v24;
    v74 = *(_DWORD *)(*v24 + 104LL);
    if ( v74 == 3 )
    {
      if ( a3 == *(_DWORD *)(v30 + 192) && *(_QWORD *)(v30 + 184) == v22 )
        return sub_2B9A8C0(a1, (char ***)v30);
    }
    else if ( v74 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
    {
      return sub_2B9A8C0(a1, (char ***)v30);
    }
    ++v24;
LABEL_153:
    v30 = *v24;
    v75 = *(_DWORD *)(*v24 + 104LL);
    if ( v75 == 3 )
    {
      if ( a3 == *(_DWORD *)(v30 + 192) && *(_QWORD *)(v30 + 184) == v22 )
        return sub_2B9A8C0(a1, (char ***)v30);
    }
    else if ( v75 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
    {
      return sub_2B9A8C0(a1, (char ***)v30);
    }
    ++v24;
    goto LABEL_157;
  }
  if ( v28 != 1 )
    goto LABEL_140;
LABEL_157:
  v30 = *v24;
  v76 = *(_DWORD *)(*v24 + 104LL);
  if ( v76 != 3 )
  {
    if ( v76 == 5 && *(_QWORD *)(v30 + 184) == v22 && a3 == *(_DWORD *)(v30 + 192) )
      return sub_2B9A8C0(a1, (char ***)v30);
LABEL_140:
    v30 = *v26;
    return sub_2B9A8C0(a1, (char ***)v30);
  }
  if ( a3 != *(_DWORD *)(v30 + 192) || *(_QWORD *)(v30 + 184) != v22 )
    goto LABEL_140;
  return sub_2B9A8C0(a1, (char ***)v30);
}
