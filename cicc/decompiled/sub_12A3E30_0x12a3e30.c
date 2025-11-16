// Function: sub_12A3E30
// Address: 0x12a3e30
//
__int64 __fastcall sub_12A3E30(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 (__fastcall **v7)(); // r14
  __int64 v8; // r13
  unsigned int v9; // eax
  _QWORD *v10; // r13
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rdx
  _BOOL8 v17; // rsi
  __int64 v18; // rdx
  __int64 result; // rax
  __int64 v20; // rdi
  bool v21; // r8
  unsigned int v22; // r15d
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  char v26; // dl
  char v27; // cl
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 (__fastcall **v38)(); // rsi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 (__fastcall **v45)(); // rsi
  __int64 (__fastcall **v46)(); // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  size_t v49; // rdx
  __int64 v50; // rcx
  char *v51; // rsi
  size_t v52; // rax
  size_t v53; // rax
  __int64 v54; // rax
  _BYTE *v55; // rdi
  size_t v56; // rdx
  __int64 v57; // rsi
  _QWORD *v58; // rax
  size_t v59; // rdx
  __int64 v60; // rdx
  int v61; // r10d
  __int64 *v62; // r9
  int v63; // ecx
  int v64; // ecx
  int v65; // eax
  int v66; // esi
  __int64 v67; // rdi
  unsigned int v68; // edx
  __int64 v69; // r8
  int v70; // r10d
  __int64 *v71; // r9
  int v72; // eax
  int v73; // edx
  __int64 v74; // rdi
  int v75; // r9d
  unsigned int v76; // r14d
  __int64 *v77; // r8
  __int64 v78; // rsi
  const char *v79; // [rsp+8h] [rbp-258h]
  __int64 v80; // [rsp+50h] [rbp-210h]
  __int64 (__fastcall **v81)(); // [rsp+50h] [rbp-210h]
  __int64 v82; // [rsp+50h] [rbp-210h]
  char v83; // [rsp+58h] [rbp-208h]
  __int64 v84; // [rsp+58h] [rbp-208h]
  __int64 (__fastcall **v85)(); // [rsp+58h] [rbp-208h]
  char *v86; // [rsp+58h] [rbp-208h]
  void *dest; // [rsp+60h] [rbp-200h] BYREF
  size_t v88; // [rsp+68h] [rbp-1F8h]
  _QWORD v89[2]; // [rsp+70h] [rbp-1F0h] BYREF
  _QWORD *v90; // [rsp+80h] [rbp-1E0h] BYREF
  size_t n; // [rsp+88h] [rbp-1D8h]
  _QWORD src[2]; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 (__fastcall **v93)(); // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v94)(); // [rsp+A8h] [rbp-1B8h]
  __int64 (__fastcall **v95)(); // [rsp+B0h] [rbp-1B0h] BYREF
  _QWORD v96[3]; // [rsp+B8h] [rbp-1A8h] BYREF
  unsigned __int64 v97; // [rsp+D0h] [rbp-190h]
  __int64 v98; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v99; // [rsp+E0h] [rbp-180h]
  __int64 v100; // [rsp+E8h] [rbp-178h]
  char v101[8]; // [rsp+F0h] [rbp-170h] BYREF
  int v102; // [rsp+F8h] [rbp-168h]
  __int64 v103[2]; // [rsp+100h] [rbp-160h] BYREF
  _QWORD v104[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v105[28]; // [rsp+120h] [rbp-140h] BYREF
  __int16 v106; // [rsp+200h] [rbp-60h]
  __int64 v107; // [rsp+208h] [rbp-58h]
  __int64 v108; // [rsp+210h] [rbp-50h]
  __int64 v109; // [rsp+218h] [rbp-48h]
  __int64 v110; // [rsp+220h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 120);
  v5 = sub_127A040(*(_QWORD *)(a1 + 32) + 8LL, v4);
  v7 = *(__int64 (__fastcall ***)())(a2 + 8);
  v8 = v5;
  if ( v7 )
    goto LABEL_2;
  LOBYTE(v89[0]) = 0;
  dest = v89;
  v88 = 0;
  sub_222DF20(v105);
  v105[27] = 0;
  v106 = 0;
  v107 = 0;
  v105[0] = off_4A06798;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v93 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v93 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v94 = 0;
  sub_222DD70((char *)&v93 + (_QWORD)*(v93 - 3), 0);
  v95 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v96[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v96[-1] + (_QWORD)*(v95 - 3), 0);
  v93 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v93 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v96[1] = 0;
  v93 = off_4A073F0;
  v105[0] = off_4A07440;
  v95 = off_4A07418;
  v96[2] = 0;
  v96[0] = off_4A07480;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  sub_220A990(v101);
  v102 = 0;
  v103[0] = (__int64)v104;
  v96[0] = off_4A07080;
  sub_12A27A0(v103, dest, (__int64)dest + v88);
  v102 = 24;
  sub_223FD50(v96, v103[0], 0, 0);
  sub_222DD70(v105, v96);
  if ( *(_QWORD *)(a1 + 160) )
  {
    v79 = *(const char **)(a1 + 160);
    v53 = strlen(v79);
    sub_223E0D0(&v95, v79, v53);
  }
  else
  {
    sub_222DC80((char *)&v96[-1] + (_QWORD)*(v95 - 3), *(_DWORD *)((char *)&v97 + (_QWORD)*(v95 - 3)) | 1u);
  }
  v54 = sub_737880(a2);
  sub_223E760(&v95, v54);
  v90 = src;
  n = 0;
  LOBYTE(src[0]) = 0;
  if ( v99 )
  {
    if ( v99 <= v97 )
      sub_2241130(&v90, 0, 0, v98, v97 - v98);
    else
      sub_2241130(&v90, 0, 0, v98, v99 - v98);
  }
  else
  {
    sub_2240AE0(&v90, v103);
  }
  v55 = dest;
  v56 = n;
  if ( v90 == src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v56 = n;
      v55 = dest;
    }
    v88 = v56;
    v55[v56] = 0;
    v55 = v90;
    goto LABEL_61;
  }
  if ( dest == v89 )
  {
    dest = v90;
    v88 = n;
    v89[0] = src[0];
    goto LABEL_99;
  }
  v57 = v89[0];
  dest = v90;
  v88 = n;
  v89[0] = src[0];
  if ( !v55 )
  {
LABEL_99:
    v90 = src;
    v55 = src;
    goto LABEL_61;
  }
  v90 = v55;
  src[0] = v57;
LABEL_61:
  n = 0;
  *v55 = 0;
  if ( v90 != src )
    j_j___libc_free_0(v90, src[0] + 1LL);
  v58 = sub_7247C0(v88 + 1);
  v59 = v88;
  *(_QWORD *)(a2 + 8) = v58;
  v4 = (__int64)v58;
  sub_2241570(&dest, v58, v59, 0);
  v60 = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(v60 + v88) = 0;
  v93 = off_4A073F0;
  v105[0] = off_4A07440;
  v95 = off_4A07418;
  v96[0] = off_4A07080;
  if ( (_QWORD *)v103[0] != v104 )
  {
    v4 = v104[0] + 1LL;
    j_j___libc_free_0(v103[0], v104[0] + 1LL);
  }
  v96[0] = off_4A07480;
  sub_2209150(v101, v4, v60);
  v93 = (__int64 (__fastcall **)())qword_4A07328;
  *(__int64 (__fastcall ***)())((char *)&v93 + qword_4A07328[-3]) = (__int64 (__fastcall **)())&unk_4A07378;
  v95 = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v96[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  v93 = (__int64 (__fastcall **)())qword_4A072D8;
  *(__int64 (__fastcall ***)())((char *)&v93 + qword_4A072D8[-3]) = (__int64 (__fastcall **)())&unk_4A07300;
  v94 = 0;
  v105[0] = off_4A06798;
  sub_222E050(v105);
  if ( dest != v89 )
  {
    v4 = v89[0] + 1LL;
    j_j___libc_free_0(dest, v89[0] + 1LL);
  }
  v7 = *(__int64 (__fastcall ***)())(a2 + 8);
LABEL_2:
  v9 = sub_127C800(a2, v4, v6);
  LOWORD(v95) = 257;
  if ( *(_BYTE *)v7 )
  {
    v93 = v7;
    LOBYTE(v95) = 3;
  }
  v10 = sub_127FC40((_QWORD *)a1, v8, (__int64)&v93, v9, 0);
  if ( sub_12A2A10(a1, a2) )
    sub_127B550("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  v11 = *(_QWORD *)(a2 + 120);
  if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 && (sub_8D4C10(v11, dword_4F077C4 != 2) & 4) != 0 )
  {
    v25 = *(_QWORD *)(a2 + 40);
    if ( v25 )
    {
      v26 = *(_BYTE *)(v25 + 28);
      if ( v26 == 17 )
      {
LABEL_36:
        if ( *((_BYTE *)v10 + 16) > 0x17u )
        {
          v32 = v10[5];
          if ( v32 )
          {
            v33 = *(_QWORD *)(v32 + 56);
            if ( v33 )
            {
              v93 = 0;
              v94 = 0;
              v95 = 0;
              v34 = sub_1649960(v33);
              v84 = v35;
              v80 = v34;
              v36 = sub_16498A0(v10);
              v37 = sub_161FF10(v36, v80, v84);
              v38 = v94;
              v90 = (_QWORD *)v37;
              if ( v94 == v95 )
              {
                sub_1273E00((__int64)&v93, v94, &v90);
              }
              else
              {
                if ( v94 )
                {
                  *v94 = (__int64 (__fastcall *)())v37;
                  v38 = v94;
                }
                v94 = v38 + 1;
              }
              v39 = sub_16498A0(v10);
              v40 = sub_1643350(v39);
              v41 = sub_159C470(v40, 0, 0);
              v44 = sub_1624210(v41, 0, v42, v43);
              v45 = v94;
              v90 = (_QWORD *)v44;
              if ( v94 == v95 )
              {
                sub_1273E00((__int64)&v93, v94, &v90);
                v46 = v94;
              }
              else
              {
                if ( v94 )
                {
                  *v94 = (__int64 (__fastcall *)())v44;
                  v45 = v94;
                }
                v46 = v45 + 1;
                v94 = v45 + 1;
              }
              v81 = v46;
              v85 = v93;
              v47 = sub_16498A0(v10);
              v48 = sub_1627350(v47, v85, v81 - v85, 0, 1);
              v49 = 0;
              v50 = v48;
              v51 = off_4CD4978[0];
              if ( off_4CD4978[0] )
              {
                v82 = v48;
                v86 = off_4CD4978[0];
                v52 = strlen(off_4CD4978[0]);
                v50 = v82;
                v51 = v86;
                v49 = v52;
              }
              sub_1626100(v10, v51, v49, v50);
              sub_1CCAB50(1, v33);
              sub_1CCABF0(3, v33);
              if ( v93 )
                j_j___libc_free_0(v93, (char *)v95 - (char *)v93);
            }
          }
        }
      }
      else if ( v26 == 2 )
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)(v25 + 16);
          if ( !v25 )
            break;
          v27 = *(_BYTE *)(v25 + 28);
          if ( v27 != 2 && v27 != 17 )
            break;
          v28 = *(_QWORD *)(v25 + 80);
          if ( !v28 )
            break;
          if ( *(_BYTE *)(v28 + 40) != 11 )
            break;
          v29 = *(_QWORD *)(v28 + 16);
          if ( v29 )
          {
            if ( *(_BYTE *)(v29 + 40) != 8 || *(_QWORD *)(v29 + 16) || *(_QWORD *)(v29 + 48) )
              break;
          }
          v30 = *(_QWORD *)(v28 + 72);
          if ( !v30 )
            break;
          if ( *(_BYTE *)(v30 + 40) != 11 )
            break;
          v31 = *(_QWORD *)(v30 + 16);
          if ( v31 )
          {
            if ( *(_BYTE *)(v31 + 40) != 8 || *(_QWORD *)(v31 + 16) || *(_QWORD *)(v31 + 48) )
              break;
          }
          if ( v27 == 17 )
            goto LABEL_36;
        }
      }
    }
  }
  v12 = *(_DWORD *)(a1 + 24);
  if ( !v12 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_85;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( a2 == *v15 )
    goto LABEL_9;
  v61 = 1;
  v62 = 0;
  while ( v16 != -8 )
  {
    if ( v16 == -16 && !v62 )
      v62 = v15;
    v14 = (v12 - 1) & (v61 + v14);
    v15 = (__int64 *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
      goto LABEL_9;
    ++v61;
  }
  v63 = *(_DWORD *)(a1 + 16);
  if ( v62 )
    v15 = v62;
  ++*(_QWORD *)a1;
  v64 = v63 + 1;
  if ( 4 * v64 >= 3 * v12 )
  {
LABEL_85:
    sub_12A2850(a1, 2 * v12);
    v65 = *(_DWORD *)(a1 + 24);
    if ( v65 )
    {
      v66 = v65 - 1;
      v67 = *(_QWORD *)(a1 + 8);
      v68 = (v65 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v64 = *(_DWORD *)(a1 + 16) + 1;
      v15 = (__int64 *)(v67 + 16LL * v68);
      v69 = *v15;
      if ( a2 != *v15 )
      {
        v70 = 1;
        v71 = 0;
        while ( v69 != -8 )
        {
          if ( !v71 && v69 == -16 )
            v71 = v15;
          v68 = v66 & (v70 + v68);
          v15 = (__int64 *)(v67 + 16LL * v68);
          v69 = *v15;
          if ( a2 == *v15 )
            goto LABEL_75;
          ++v70;
        }
        if ( v71 )
          v15 = v71;
      }
      goto LABEL_75;
    }
    goto LABEL_119;
  }
  if ( v12 - *(_DWORD *)(a1 + 20) - v64 <= v12 >> 3 )
  {
    sub_12A2850(a1, v12);
    v72 = *(_DWORD *)(a1 + 24);
    if ( v72 )
    {
      v73 = v72 - 1;
      v74 = *(_QWORD *)(a1 + 8);
      v75 = 1;
      v76 = (v72 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v77 = 0;
      v64 = *(_DWORD *)(a1 + 16) + 1;
      v15 = (__int64 *)(v74 + 16LL * v76);
      v78 = *v15;
      if ( a2 != *v15 )
      {
        while ( v78 != -8 )
        {
          if ( !v77 && v78 == -16 )
            v77 = v15;
          v76 = v73 & (v75 + v76);
          v15 = (__int64 *)(v74 + 16LL * v76);
          v78 = *v15;
          if ( a2 == *v15 )
            goto LABEL_75;
          ++v75;
        }
        if ( v77 )
          v15 = v77;
      }
      goto LABEL_75;
    }
LABEL_119:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_75:
  *(_DWORD *)(a1 + 16) = v64;
  if ( *v15 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v15 = a2;
  v15[1] = 0;
LABEL_9:
  v15[1] = (__int64)v10;
  v17 = 0;
  sub_72F9F0(a2, 0, &v90, &v93);
  if ( (_BYTE)v90 == 3 )
  {
    v20 = *(_QWORD *)(a2 + 120);
    v21 = 0;
    if ( (*(_BYTE *)(v20 + 140) & 0xFB) == 8 )
    {
      v17 = dword_4F077C4 != 2;
      v21 = (sub_8D4C10(v20, v17) & 2) != 0;
    }
    v83 = v21;
    v22 = sub_127C800(a2, v17, v18);
    v23 = sub_12A2A10(a1, a2);
    v24 = sub_127D2A0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 120));
    sub_1280F50((__int64 *)a1, v24, v23, v22, v83);
  }
  result = dword_4D046B4;
  if ( dword_4D046B4 )
  {
    if ( (*(_BYTE *)(a2 + 174) & 4) == 0 )
      return sub_12A2480(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 384LL), a2, (__int64)v10, a1 + 48);
  }
  return result;
}
