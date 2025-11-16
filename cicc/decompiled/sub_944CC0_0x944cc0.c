// Function: sub_944CC0
// Address: 0x944cc0
//
__int64 __fastcall sub_944CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 (__fastcall **v9)(); // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int16 v12; // cx
  __int64 v13; // r13
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 v16; // rdi
  int v17; // r11d
  __int64 *v18; // r9
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rax
  _BOOL8 v23; // rsi
  __int64 v24; // rdx
  __int64 result; // rax
  __int64 v26; // rdi
  bool v27; // r8
  unsigned int v28; // r15d
  unsigned __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rax
  char v35; // dl
  char v36; // cl
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 (__fastcall **v47)(); // rsi
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 (__fastcall **v54)(); // rsi
  __int64 (__fastcall **v55)(); // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  size_t v58; // rdx
  __int64 v59; // rcx
  char *v60; // rsi
  size_t v61; // rax
  int v62; // eax
  int v63; // edx
  size_t v64; // rax
  __int64 v65; // rax
  _BYTE *v66; // rdi
  size_t v67; // rdx
  __int64 v68; // rsi
  _QWORD *v69; // rax
  size_t v70; // rdx
  __int64 v71; // rdx
  int v72; // eax
  int v73; // esi
  __int64 v74; // rdi
  unsigned int v75; // ecx
  __int64 v76; // rax
  int v77; // r10d
  __int64 *v78; // r8
  int v79; // eax
  int v80; // ecx
  __int64 v81; // rsi
  int v82; // r8d
  __int64 v83; // r14
  __int64 *v84; // rdi
  __int64 v85; // rax
  const char *v86; // [rsp+8h] [rbp-258h]
  __int64 v87; // [rsp+50h] [rbp-210h]
  __int64 (__fastcall **v88)(); // [rsp+50h] [rbp-210h]
  __int64 v89; // [rsp+50h] [rbp-210h]
  char v90; // [rsp+58h] [rbp-208h]
  __int64 v91; // [rsp+58h] [rbp-208h]
  __int64 (__fastcall **v92)(); // [rsp+58h] [rbp-208h]
  char *v93; // [rsp+58h] [rbp-208h]
  void *dest; // [rsp+60h] [rbp-200h] BYREF
  size_t v95; // [rsp+68h] [rbp-1F8h]
  _QWORD v96[2]; // [rsp+70h] [rbp-1F0h] BYREF
  _QWORD *v97; // [rsp+80h] [rbp-1E0h] BYREF
  size_t n; // [rsp+88h] [rbp-1D8h]
  _QWORD src[2]; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 (__fastcall **v100)(); // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v101)(); // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v102)(); // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 (__fastcall **v103)(); // [rsp+B8h] [rbp-1A8h] BYREF
  __int64 v104; // [rsp+C0h] [rbp-1A0h]
  __int64 v105; // [rsp+C8h] [rbp-198h]
  unsigned __int64 v106; // [rsp+D0h] [rbp-190h]
  __int64 v107; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v108; // [rsp+E0h] [rbp-180h]
  __int64 v109; // [rsp+E8h] [rbp-178h]
  char v110[8]; // [rsp+F0h] [rbp-170h] BYREF
  int v111; // [rsp+F8h] [rbp-168h]
  __int64 v112[2]; // [rsp+100h] [rbp-160h] BYREF
  _QWORD v113[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v114[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v115; // [rsp+200h] [rbp-60h]
  __int64 v116; // [rsp+208h] [rbp-58h]
  __int64 v117; // [rsp+210h] [rbp-50h]
  __int64 v118; // [rsp+218h] [rbp-48h]
  __int64 v119; // [rsp+220h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 120);
  v7 = sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, v6, a3, a4);
  v9 = *(__int64 (__fastcall ***)())(a2 + 8);
  v10 = v7;
  if ( v9 )
    goto LABEL_2;
  LOBYTE(v96[0]) = 0;
  dest = v96;
  v95 = 0;
  sub_222DF20(v114);
  v114[27] = 0;
  v116 = 0;
  v117 = 0;
  v114[0] = off_4A06798;
  v115 = 0;
  v118 = 0;
  v119 = 0;
  v100 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v100 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v101 = 0;
  sub_222DD70((char *)&v100 + (_QWORD)*(v100 - 3), 0);
  v102 = (__int64 (__fastcall **)())qword_4A07288;
  *(__int64 (__fastcall ***)())((char *)&v102 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
  sub_222DD70((char *)&v102 + (_QWORD)*(v102 - 3), 0);
  v100 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v100 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v100 = off_4A073F0;
  v114[0] = off_4A07440;
  v102 = off_4A07418;
  v103 = off_4A07480;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  sub_220A990(v110);
  v112[0] = (__int64)v113;
  v103 = off_4A07080;
  v111 = 0;
  sub_943740(v112, dest, (__int64)dest + v95);
  v111 = 24;
  sub_223FD50(&v103, v112[0], 0, 0);
  sub_222DD70(v114, &v103);
  if ( *(_QWORD *)(a1 + 232) )
  {
    v86 = *(const char **)(a1 + 232);
    v64 = strlen(v86);
    sub_223E0D0(&v102, v86, v64);
  }
  else
  {
    sub_222DC80((char *)&v102 + (_QWORD)*(v102 - 3), *(_DWORD *)((char *)&v106 + (_QWORD)*(v102 - 3)) | 1u);
  }
  v65 = sub_737880(a2);
  sub_223E760(&v102, v65);
  v97 = src;
  n = 0;
  LOBYTE(src[0]) = 0;
  if ( v108 )
  {
    if ( v108 <= v106 )
      sub_2241130(&v97, 0, 0, v107, v106 - v107);
    else
      sub_2241130(&v97, 0, 0, v107, v108 - v107);
  }
  else
  {
    sub_2240AE0(&v97, v112);
  }
  v66 = dest;
  v67 = n;
  if ( v97 == src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v67 = n;
      v66 = dest;
    }
    v95 = v67;
    v66[v67] = 0;
    v66 = v97;
    goto LABEL_77;
  }
  if ( dest == v96 )
  {
    dest = v97;
    v95 = n;
    v96[0] = src[0];
    goto LABEL_106;
  }
  v68 = v96[0];
  dest = v97;
  v95 = n;
  v96[0] = src[0];
  if ( !v66 )
  {
LABEL_106:
    v97 = src;
    v66 = src;
    goto LABEL_77;
  }
  v97 = v66;
  src[0] = v68;
LABEL_77:
  n = 0;
  *v66 = 0;
  if ( v97 != src )
    j_j___libc_free_0(v97, src[0] + 1LL);
  v69 = sub_7247C0(v95 + 1);
  v70 = v95;
  *(_QWORD *)(a2 + 8) = v69;
  v6 = (__int64)v69;
  sub_2241570(&dest, v69, v70, 0);
  v71 = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(v71 + v95) = 0;
  v100 = off_4A073F0;
  v114[0] = off_4A07440;
  v102 = off_4A07418;
  v103 = off_4A07080;
  if ( (_QWORD *)v112[0] != v113 )
  {
    v6 = v113[0] + 1LL;
    j_j___libc_free_0(v112[0], v113[0] + 1LL);
  }
  v103 = off_4A07480;
  sub_2209150(v110, v6, v71);
  v100 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v100 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v102 = (__int64 (__fastcall **)())qword_4A07288;
  *(__int64 (__fastcall ***)())((char *)&v102 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
  v100 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v100 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v101 = 0;
  v114[0] = off_4A06798;
  sub_222E050(v114);
  if ( dest != v96 )
  {
    v6 = v96[0] + 1LL;
    j_j___libc_free_0(dest, v96[0] + 1LL);
  }
  v9 = *(__int64 (__fastcall ***)())(a2 + 8);
LABEL_2:
  LODWORD(v11) = sub_91CB50(a2, v6, v8);
  v12 = 0;
  if ( (_DWORD)v11 )
  {
    _BitScanReverse64((unsigned __int64 *)&v11, (unsigned int)v11);
    LOBYTE(v12) = 63 - (v11 ^ 0x3F);
    HIBYTE(v12) = 1;
  }
  LOWORD(v104) = 257;
  if ( *(_BYTE *)v9 )
  {
    v100 = v9;
    LOBYTE(v104) = 3;
  }
  v13 = sub_921B80(a1, v10, (__int64)&v100, v12, 0);
  if ( sub_9439D0(a1, a2) )
    sub_91B8A0("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  v14 = *(_QWORD *)(a2 + 120);
  if ( (*(_BYTE *)(v14 + 140) & 0xFB) == 8 )
  {
    v33 = dword_4F077C4 != 2;
    if ( (sub_8D4C10(v14, v33) & 4) != 0 )
    {
      v34 = *(_QWORD *)(a2 + 40);
      if ( v34 )
      {
        v35 = *(_BYTE *)(v34 + 28);
        if ( v35 == 17 )
        {
LABEL_39:
          if ( *(_BYTE *)v13 > 0x1Cu )
          {
            v40 = *(_QWORD *)(v13 + 40);
            if ( v40 )
            {
              v41 = *(_QWORD *)(v40 + 72);
              if ( v41 )
              {
                v100 = 0;
                v101 = 0;
                v102 = 0;
                v42 = sub_BD5D20(v41);
                v91 = v43;
                v87 = v42;
                v44 = sub_BD5C60(v13, v33, v43);
                v45 = sub_B9B140(v44, v87, v91);
                v47 = v101;
                v97 = (_QWORD *)v45;
                if ( v101 == v102 )
                {
                  sub_914280((__int64)&v100, v101, &v97);
                }
                else
                {
                  if ( v101 )
                  {
                    *v101 = (__int64 (__fastcall *)())v45;
                    v47 = v101;
                  }
                  v101 = ++v47;
                }
                v48 = sub_BD5C60(v13, v47, v46);
                v49 = sub_BCB2D0(v48);
                v50 = sub_ACD640(v49, 0, 0);
                v53 = sub_B98A20(v50, 0, v51, v52);
                v54 = v101;
                v97 = (_QWORD *)v53;
                if ( v101 == v102 )
                {
                  sub_914280((__int64)&v100, v101, &v97);
                  v55 = v101;
                }
                else
                {
                  if ( v101 )
                  {
                    *v101 = (__int64 (__fastcall *)())v53;
                    v54 = v101;
                  }
                  v55 = v54 + 1;
                  v101 = v54 + 1;
                }
                v88 = v55;
                v92 = v100;
                v56 = sub_BD5C60(v13, v54, v55);
                v57 = sub_B9C770(v56, v92, v88 - v92, 0, 1);
                v58 = 0;
                v59 = v57;
                v60 = off_4C5D0D8[0];
                if ( off_4C5D0D8[0] )
                {
                  v89 = v57;
                  v93 = off_4C5D0D8[0];
                  v61 = strlen(off_4C5D0D8[0]);
                  v59 = v89;
                  v60 = v93;
                  v58 = v61;
                }
                sub_B9A090(v13, v60, v58, v59);
                sub_CEF870(1, v41);
                sub_CEF900(3, v41);
                if ( v100 )
                  j_j___libc_free_0(v100, (char *)v102 - (char *)v100);
              }
            }
          }
        }
        else if ( v35 == 2 )
        {
          while ( 1 )
          {
            v34 = *(_QWORD *)(v34 + 16);
            if ( !v34 )
              break;
            v36 = *(_BYTE *)(v34 + 28);
            if ( v36 != 2 && v36 != 17 )
              break;
            v37 = *(_QWORD *)(v34 + 80);
            if ( !v37 )
              break;
            if ( *(_BYTE *)(v37 + 40) != 11 )
              break;
            v33 = *(_QWORD *)(v37 + 16);
            if ( v33 )
            {
              if ( *(_BYTE *)(v33 + 40) != 8 || *(_QWORD *)(v33 + 16) || *(_QWORD *)(v33 + 48) )
                break;
            }
            v38 = *(_QWORD *)(v37 + 72);
            if ( !v38 )
              break;
            if ( *(_BYTE *)(v38 + 40) != 11 )
              break;
            v39 = *(_QWORD *)(v38 + 16);
            if ( v39 )
            {
              if ( *(_BYTE *)(v39 + 40) != 8 || *(_QWORD *)(v39 + 16) || *(_QWORD *)(v39 + 48) )
                break;
            }
            if ( v36 == 17 )
              goto LABEL_39;
          }
        }
      }
    }
  }
  v15 = *(_DWORD *)(a1 + 24);
  if ( !v15 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_92;
  }
  v16 = *(_QWORD *)(a1 + 8);
  v17 = 1;
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v16 + 16LL * v19);
  v21 = *v20;
  if ( a2 == *v20 )
  {
LABEL_11:
    v22 = v20 + 1;
    goto LABEL_12;
  }
  while ( v21 != -4096 )
  {
    if ( v21 == -8192 && !v18 )
      v18 = v20;
    v19 = (v15 - 1) & (v17 + v19);
    v20 = (__int64 *)(v16 + 16LL * v19);
    v21 = *v20;
    if ( a2 == *v20 )
      goto LABEL_11;
    ++v17;
  }
  if ( !v18 )
    v18 = v20;
  v62 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v63 = v62 + 1;
  if ( 4 * (v62 + 1) >= 3 * v15 )
  {
LABEL_92:
    sub_9437F0(a1, 2 * v15);
    v72 = *(_DWORD *)(a1 + 24);
    if ( v72 )
    {
      v73 = v72 - 1;
      v74 = *(_QWORD *)(a1 + 8);
      v75 = (v72 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v63 = *(_DWORD *)(a1 + 16) + 1;
      v18 = (__int64 *)(v74 + 16LL * v75);
      v76 = *v18;
      if ( a2 != *v18 )
      {
        v77 = 1;
        v78 = 0;
        while ( v76 != -4096 )
        {
          if ( !v78 && v76 == -8192 )
            v78 = v18;
          v75 = v73 & (v77 + v75);
          v18 = (__int64 *)(v74 + 16LL * v75);
          v76 = *v18;
          if ( a2 == *v18 )
            goto LABEL_65;
          ++v77;
        }
        if ( v78 )
          v18 = v78;
      }
      goto LABEL_65;
    }
    goto LABEL_121;
  }
  if ( v15 - *(_DWORD *)(a1 + 20) - v63 <= v15 >> 3 )
  {
    sub_9437F0(a1, v15);
    v79 = *(_DWORD *)(a1 + 24);
    if ( v79 )
    {
      v80 = v79 - 1;
      v81 = *(_QWORD *)(a1 + 8);
      v82 = 1;
      LODWORD(v83) = (v79 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v84 = 0;
      v63 = *(_DWORD *)(a1 + 16) + 1;
      v18 = (__int64 *)(v81 + 16LL * (unsigned int)v83);
      v85 = *v18;
      if ( a2 != *v18 )
      {
        while ( v85 != -4096 )
        {
          if ( !v84 && v85 == -8192 )
            v84 = v18;
          v83 = v80 & (unsigned int)(v83 + v82);
          v18 = (__int64 *)(v81 + 16 * v83);
          v85 = *v18;
          if ( a2 == *v18 )
            goto LABEL_65;
          ++v82;
        }
        if ( v84 )
          v18 = v84;
      }
      goto LABEL_65;
    }
LABEL_121:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_65:
  *(_DWORD *)(a1 + 16) = v63;
  if ( *v18 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v18 = a2;
  v22 = v18 + 1;
  v18[1] = 0;
LABEL_12:
  *v22 = v13;
  v23 = 0;
  sub_72F9F0(a2, 0, &v97, &v100);
  if ( (_BYTE)v97 == 3 )
  {
    v26 = *(_QWORD *)(a2 + 120);
    v27 = 0;
    if ( (*(_BYTE *)(v26 + 140) & 0xFB) == 8 )
    {
      v23 = dword_4F077C4 != 2;
      v27 = (sub_8D4C10(v26, v23) & 2) != 0;
    }
    v90 = v27;
    v28 = sub_91CB50(a2, v23, v24);
    v29 = sub_9439D0(a1, a2);
    v32 = sub_91DAD0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 120), v30, v31);
    sub_923130(a1, v32, v29, v28, v90);
  }
  result = dword_4D046B4;
  if ( dword_4D046B4 )
  {
    if ( (*(_BYTE *)(a2 + 174) & 4) == 0 )
      return sub_943410(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 368LL), a2, v13, a1 + 48);
  }
  return result;
}
