// Function: sub_1C43660
// Address: 0x1c43660
//
__int64 __fastcall sub_1C43660(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned __int8 v5; // al
  unsigned int v6; // r15d
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  const char *v17; // r12
  const char *v18; // rdx
  const char *v19; // r13
  char *v20; // rdi
  size_t v21; // rcx
  __int64 (__fastcall **v22)(); // rdx
  __int64 (__fastcall **v23)(); // rsi
  _QWORD *v24; // rax
  unsigned int v25; // r13d
  __int64 v26; // rcx
  char v27; // al
  __int64 (__fastcall ***v28)(); // rdi
  __int64 v29; // r13
  unsigned int v30; // eax
  __int64 v31; // r13
  int v32; // r14d
  _QWORD *v33; // r15
  _QWORD *v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r9
  const char *v38; // r10
  size_t v39; // r8
  size_t v40; // rax
  const char *v41; // rdx
  __int64 v42; // rax
  char *v43; // rsi
  size_t v44; // rdx
  char *v45; // rdi
  __int64 v46; // rcx
  const char *v47; // rax
  char *v48; // rdi
  const char *v49; // rdx
  const char *v50; // r14
  size_t v51; // r13
  char *v52; // rdi
  __int64 (__fastcall **v53)(); // rdx
  size_t v54; // rcx
  __int64 (__fastcall **v55)(); // rsi
  size_t v56; // rdx
  __int64 (__fastcall ***v57)(); // rdi
  size_t v58; // rdx
  const char *v59; // rdx
  __int64 (__fastcall ***v60)(); // rdi
  size_t v61; // [rsp+0h] [rbp-250h]
  const char *v62; // [rsp+8h] [rbp-248h]
  __int64 v63; // [rsp+10h] [rbp-240h]
  int v64; // [rsp+48h] [rbp-208h]
  void *dest; // [rsp+50h] [rbp-200h]
  size_t v67; // [rsp+68h] [rbp-1E8h] BYREF
  __int64 (__fastcall **v68)(); // [rsp+70h] [rbp-1E0h] BYREF
  size_t n; // [rsp+78h] [rbp-1D8h]
  char src[16]; // [rsp+80h] [rbp-1D0h] BYREF
  void *s1; // [rsp+90h] [rbp-1C0h] BYREF
  size_t v72; // [rsp+98h] [rbp-1B8h]
  __int64 (__fastcall **v73)(); // [rsp+A0h] [rbp-1B0h] BYREF
  _QWORD v74[3]; // [rsp+A8h] [rbp-1A8h] BYREF
  unsigned __int64 v75; // [rsp+C0h] [rbp-190h]
  __int64 v76; // [rsp+C8h] [rbp-188h]
  unsigned __int64 v77; // [rsp+D0h] [rbp-180h]
  __int64 v78; // [rsp+D8h] [rbp-178h]
  _BYTE v79[8]; // [rsp+E0h] [rbp-170h] BYREF
  int v80; // [rsp+E8h] [rbp-168h]
  _QWORD v81[2]; // [rsp+F0h] [rbp-160h] BYREF
  _QWORD v82[2]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD v83[28]; // [rsp+110h] [rbp-140h] BYREF
  __int16 v84; // [rsp+1F0h] [rbp-60h]
  __int64 v85; // [rsp+1F8h] [rbp-58h]
  __int64 v86; // [rsp+200h] [rbp-50h]
  __int64 v87; // [rsp+208h] [rbp-48h]
  __int64 v88; // [rsp+210h] [rbp-40h]

  v3 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 <= 0x17u )
    goto LABEL_5;
  while ( 1 )
  {
    if ( v5 == 78 )
    {
      v13 = *(_QWORD *)(v3 - 24);
      if ( !*(_BYTE *)(v13 + 16) && (*(_BYTE *)(v13 + 33) & 0x20) != 0 && *(_DWORD *)(v13 + 36) == 5232 )
      {
        v14 = *(unsigned __int8 **)(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)) + 24LL);
        a2 = *v14;
        if ( (unsigned __int8)(a2 - 4) <= 0x1Eu )
        {
          v15 = *(_QWORD *)&v14[-8 * *((unsigned int *)v14 + 2)];
          if ( *(_BYTE *)v15 == 1 )
          {
            v16 = *(_QWORD *)(v15 + 136);
            if ( *(_BYTE *)(v16 + 16) == 3 )
            {
              if ( (unsigned __int8)sub_1C2E830(*(_QWORD *)(v15 + 136)) )
              {
                v17 = sub_1CCA9F0(v16);
                v19 = v18;
              }
              else
              {
                if ( (unsigned __int8)sub_1C2E860(v16) )
                  v17 = sub_1CCAA00(v16);
                else
                  v17 = sub_1CCAA10(v16);
                v19 = v59;
              }
              if ( !v17 )
              {
                LOBYTE(v73) = 0;
                v58 = 0;
                s1 = &v73;
                v20 = *(char **)a3;
                goto LABEL_127;
              }
              v68 = (__int64 (__fastcall **)())v19;
              s1 = &v73;
              if ( (unsigned __int64)v19 > 0xF )
              {
                s1 = (void *)sub_22409D0(&s1, &v68, 0);
                v60 = (__int64 (__fastcall ***)())s1;
                v73 = v68;
              }
              else
              {
                if ( v19 == (const char *)1 )
                {
                  LOBYTE(v73) = *v17;
LABEL_27:
                  v72 = (size_t)v68;
                  *((_BYTE *)v68 + (_QWORD)s1) = 0;
                  v20 = *(char **)a3;
                  if ( s1 != &v73 )
                  {
                    v21 = v72;
                    v22 = v73;
                    if ( v20 == (char *)(a3 + 16) )
                    {
                      *(_QWORD *)a3 = s1;
                      *(_QWORD *)(a3 + 8) = v21;
                      *(_QWORD *)(a3 + 16) = v22;
                    }
                    else
                    {
                      v23 = *(__int64 (__fastcall ***)())(a3 + 16);
                      *(_QWORD *)a3 = s1;
                      *(_QWORD *)(a3 + 8) = v21;
                      *(_QWORD *)(a3 + 16) = v22;
                      if ( v20 )
                      {
                        s1 = v20;
                        v73 = v23;
LABEL_31:
                        v6 = 1;
                        v72 = 0;
                        *(_BYTE *)s1 = 0;
                        sub_2240A30(&s1);
                        return v6;
                      }
                    }
                    s1 = &v73;
                    goto LABEL_31;
                  }
                  v58 = v72;
                  if ( v72 )
                  {
                    if ( v72 == 1 )
                      *v20 = (char)v73;
                    else
                      memcpy(v20, &v73, v72);
                    v58 = v72;
                    v20 = *(char **)a3;
                  }
LABEL_127:
                  *(_QWORD *)(a3 + 8) = v58;
                  v20[v58] = 0;
                  goto LABEL_31;
                }
                if ( !v19 )
                  goto LABEL_27;
                v60 = &v73;
              }
              memcpy(v60, v17, (size_t)v19);
              goto LABEL_27;
            }
          }
        }
      }
LABEL_5:
      v6 = sub_1CCAE30(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 40LL));
      if ( !(_BYTE)v6 )
        return 0;
      v7 = *(_QWORD *)(a1 + 240);
      if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
      {
        sub_15E08E0(*(_QWORD *)(a1 + 240), a2);
        v8 = *(_QWORD *)(v7 + 88);
        v7 = *(_QWORD *)(a1 + 240);
        if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
          sub_15E08E0(*(_QWORD *)(a1 + 240), a2);
        v9 = *(_QWORD *)(v7 + 88);
      }
      else
      {
        v8 = *(_QWORD *)(v7 + 88);
        v9 = v8;
      }
      v10 = v9 + 40LL * *(_QWORD *)(v7 + 96);
      if ( v10 == v8 )
        return 0;
      if ( v3 != v8 )
      {
        LODWORD(v11) = 0;
        while ( 1 )
        {
          v8 += 40;
          v11 = (unsigned int)(v11 + 1);
          if ( v8 == v10 )
            return 0;
          if ( v8 == v3 )
          {
            v63 = v11;
            goto LABEL_55;
          }
        }
      }
      v63 = 0;
LABEL_55:
      sub_222DF20(v83);
      v83[27] = 0;
      v85 = 0;
      v86 = 0;
      v83[0] = off_4A06798;
      v84 = 0;
      v87 = 0;
      v88 = 0;
      s1 = qword_4A072D8;
      *(void **)((char *)&s1 + qword_4A072D8[-3]) = &unk_4A07300;
      v72 = 0;
      sub_222DD70((char *)&s1 + *((_QWORD *)s1 - 3), 0);
      v73 = (__int64 (__fastcall **)())qword_4A07288;
      *(_QWORD *)((char *)&v74[-1] + qword_4A07288[-3]) = &unk_4A072B0;
      sub_222DD70((char *)&v74[-1] + (_QWORD)*(v73 - 3), 0);
      s1 = qword_4A07328;
      *(void **)((char *)&s1 + qword_4A07328[-3]) = &unk_4A07378;
      v74[1] = 0;
      v74[2] = 0;
      s1 = off_4A073F0;
      v75 = 0;
      v76 = 0;
      v83[0] = off_4A07440;
      v77 = 0;
      v78 = 0;
      v73 = off_4A07418;
      v74[0] = off_4A07480;
      sub_220A990(v79);
      v80 = 24;
      LOBYTE(v82[0]) = 0;
      v74[0] = off_4A07080;
      v81[0] = v82;
      v81[1] = 0;
      sub_222DD70(v83, v74);
      v38 = sub_1649960(*(_QWORD *)(a1 + 240));
      v39 = v35;
      if ( !v38 )
      {
        src[0] = 0;
        v68 = (__int64 (__fastcall **)())src;
        n = 0;
        v42 = sub_223E0D0(&v73, src, 0, v36, v35, v37);
LABEL_64:
        dest = (void *)v42;
        sub_223E0D0(v42, "_param_", 7);
        sub_223E760(dest, v63);
        if ( v68 != (__int64 (__fastcall **)())src )
          j_j___libc_free_0(v68, *(_QWORD *)src + 1LL);
        v68 = (__int64 (__fastcall **)())src;
        n = 0;
        src[0] = 0;
        if ( v77 )
        {
          v43 = 0;
          if ( v77 <= v75 )
            sub_2241130(&v68, 0, 0, v76, v75 - v76);
          else
            sub_2241130(&v68, 0, 0, v76, v77 - v76);
        }
        else
        {
          v43 = (char *)v81;
          sub_2240AE0(&v68, v81);
        }
        v44 = n;
        v45 = *(char **)a3;
        if ( v68 == (__int64 (__fastcall **)())src )
        {
          if ( n )
          {
            if ( n == 1 )
            {
              *v45 = src[0];
            }
            else
            {
              v43 = src;
              memcpy(v45, src, n);
            }
            v44 = n;
            v45 = *(char **)a3;
          }
          *(_QWORD *)(a3 + 8) = v44;
          v45[v44] = 0;
          v45 = (char *)v68;
          goto LABEL_73;
        }
        v43 = *(char **)src;
        if ( v45 == (char *)(a3 + 16) )
        {
          *(_QWORD *)a3 = v68;
          *(_QWORD *)(a3 + 8) = v44;
          *(_QWORD *)(a3 + 16) = v43;
        }
        else
        {
          v46 = *(_QWORD *)(a3 + 16);
          *(_QWORD *)a3 = v68;
          *(_QWORD *)(a3 + 8) = v44;
          *(_QWORD *)(a3 + 16) = v43;
          if ( v45 )
          {
            v68 = (__int64 (__fastcall **)())v45;
            *(_QWORD *)src = v46;
LABEL_73:
            n = 0;
            *v45 = 0;
            if ( v68 != (__int64 (__fastcall **)())src )
            {
              v43 = (char *)(*(_QWORD *)src + 1LL);
              j_j___libc_free_0(v68, *(_QWORD *)src + 1LL);
            }
            s1 = off_4A073F0;
            v83[0] = off_4A07440;
            v73 = off_4A07418;
            v74[0] = off_4A07080;
            if ( (_QWORD *)v81[0] != v82 )
            {
              v43 = (char *)(v82[0] + 1LL);
              j_j___libc_free_0(v81[0], v82[0] + 1LL);
            }
            v74[0] = off_4A07480;
            sub_2209150(v79, v43, v44);
            s1 = qword_4A07328;
            *(void **)((char *)&s1 + qword_4A07328[-3]) = &unk_4A07378;
            v73 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v74[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            s1 = qword_4A072D8;
            *(void **)((char *)&s1 + qword_4A072D8[-3]) = &unk_4A07300;
            v72 = 0;
            v83[0] = off_4A06798;
            sub_222E050(v83);
            return v6;
          }
        }
        v68 = (__int64 (__fastcall **)())src;
        v45 = src;
        goto LABEL_73;
      }
      v67 = v35;
      v40 = v35;
      v68 = (__int64 (__fastcall **)())src;
      if ( v35 > 0xF )
      {
        v61 = v35;
        v62 = v38;
        v47 = (const char *)sub_22409D0(&v68, &v67, 0);
        v38 = v62;
        v39 = v61;
        v68 = (__int64 (__fastcall **)())v47;
        v48 = (char *)v47;
        *(_QWORD *)src = v67;
      }
      else
      {
        if ( v35 == 1 )
        {
          src[0] = *v38;
          v41 = src;
LABEL_59:
          n = v40;
          v41[v40] = 0;
          v42 = sub_223E0D0(&v73, (const char *)v68, n, v36, v39, v37, v61, v62);
          goto LABEL_64;
        }
        if ( !v35 )
        {
          v41 = src;
          goto LABEL_59;
        }
        v48 = src;
      }
      memcpy(v48, v38, v39);
      v40 = v67;
      v41 = (const char *)v68;
      goto LABEL_59;
    }
    if ( v5 != 54 )
    {
      if ( v5 != 77 )
        goto LABEL_5;
      if ( (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) != 0 )
      {
        v24 = (*(_BYTE *)(v3 + 23) & 0x40) != 0
            ? *(_QWORD **)(v3 - 8)
            : (_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
        v6 = sub_1C43660(a1, *v24, a3);
        if ( (_BYTE)v6 )
        {
          v25 = 1;
          v64 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
          if ( v64 == 1 )
            return v6;
          while ( 1 )
          {
            LOBYTE(v73) = 0;
            v72 = 0;
            s1 = &v73;
            v26 = (*(_BYTE *)(v3 + 23) & 0x40) != 0
                ? *(_QWORD *)(v3 - 8)
                : v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
            v27 = sub_1C43660(a1, *(_QWORD *)(v26 + 24LL * v25), &s1);
            v28 = (__int64 (__fastcall ***)())s1;
            if ( !v27 )
              break;
            if ( v72 != *(_QWORD *)(a3 + 8) )
              break;
            if ( v72 )
            {
              v28 = (__int64 (__fastcall ***)())s1;
              if ( memcmp(s1, *(const void **)a3, v72) )
                break;
            }
            if ( v28 != &v73 )
              j_j___libc_free_0(v28, (char *)v73 + 1);
            if ( ++v25 == v64 )
              return (unsigned __int8)v6;
          }
          if ( v28 != &v73 )
            j_j___libc_free_0(v28, (char *)v73 + 1);
        }
      }
      return 0;
    }
    v29 = *(_QWORD *)(v3 - 24);
    if ( !v29 )
      BUG();
    if ( *(_BYTE *)(v29 + 16) <= 3u )
    {
      v30 = sub_1C2E890(*(_QWORD *)(v3 - 24));
      if ( (_BYTE)v30 )
        break;
    }
    v31 = *(_QWORD *)(v29 + 8);
    if ( v31 )
    {
      v32 = 0;
      v33 = 0;
      do
      {
        while ( 1 )
        {
          v34 = sub_1648700(v31);
          if ( *((_BYTE *)v34 + 16) == 55 )
            break;
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            goto LABEL_51;
        }
        v31 = *(_QWORD *)(v31 + 8);
        ++v32;
        v33 = v34;
      }
      while ( v31 );
LABEL_51:
      if ( v32 == 1 )
      {
        v3 = *(v33 - 6);
        v5 = *(_BYTE *)(v3 + 16);
        if ( v5 > 0x17u )
          continue;
      }
    }
    goto LABEL_5;
  }
  v6 = v30;
  v50 = sub_1CCAA10(v29);
  v51 = (size_t)v49;
  if ( !v50 )
  {
    LOBYTE(v73) = 0;
    v56 = 0;
    s1 = &v73;
    v52 = *(char **)a3;
LABEL_113:
    *(_QWORD *)(a3 + 8) = v56;
    v52[v56] = 0;
    goto LABEL_110;
  }
  v68 = (__int64 (__fastcall **)())v49;
  s1 = &v73;
  if ( (unsigned __int64)v49 > 0xF )
  {
    s1 = (void *)sub_22409D0(&s1, &v68, 0);
    v57 = (__int64 (__fastcall ***)())s1;
    v73 = v68;
LABEL_122:
    memcpy(v57, v50, v51);
  }
  else
  {
    if ( v49 == (const char *)1 )
    {
      LOBYTE(v73) = *v50;
      goto LABEL_106;
    }
    if ( v49 )
    {
      v57 = &v73;
      goto LABEL_122;
    }
  }
LABEL_106:
  v72 = (size_t)v68;
  *((_BYTE *)v68 + (_QWORD)s1) = 0;
  v52 = *(char **)a3;
  if ( s1 == &v73 )
  {
    v56 = v72;
    if ( v72 )
    {
      if ( v72 == 1 )
        *v52 = (char)v73;
      else
        memcpy(v52, &v73, v72);
      v56 = v72;
      v52 = *(char **)a3;
    }
    goto LABEL_113;
  }
  v53 = v73;
  v54 = v72;
  if ( v52 == (char *)(a3 + 16) )
  {
    *(_QWORD *)a3 = s1;
    *(_QWORD *)(a3 + 8) = v54;
    *(_QWORD *)(a3 + 16) = v53;
    goto LABEL_115;
  }
  v55 = *(__int64 (__fastcall ***)())(a3 + 16);
  *(_QWORD *)a3 = s1;
  *(_QWORD *)(a3 + 8) = v54;
  *(_QWORD *)(a3 + 16) = v53;
  if ( !v52 )
  {
LABEL_115:
    s1 = &v73;
    goto LABEL_110;
  }
  s1 = v52;
  v73 = v55;
LABEL_110:
  v72 = 0;
  *(_BYTE *)s1 = 0;
  if ( s1 != &v73 )
    j_j___libc_free_0(s1, (char *)v73 + 1);
  return v6;
}
