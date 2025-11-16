// Function: sub_1A13400
// Address: 0x1a13400
//
__int64 __fastcall sub_1A13400(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // r10
  unsigned int v15; // ecx
  __int64 v16; // rsi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  unsigned int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // r12
  int v25; // r9d
  char *v26; // rbx
  int v27; // r12d
  _QWORD *v28; // r10
  char *v29; // r15
  char *v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rcx
  int v33; // r11d
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rsi
  unsigned int i; // eax
  __int64 v37; // r13
  unsigned int v38; // eax
  char *v39; // r13
  __int64 v40; // rdx
  char *v41; // rax
  bool v42; // zf
  __int64 v43; // rbx
  __int64 v44; // rbx
  __int64 v45; // r14
  __int64 v46; // r12
  unsigned __int64 v47; // rax
  _BYTE *v48; // rsi
  __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // rsi
  __int64 *v52; // rax
  signed __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rdx
  bool v56; // cf
  unsigned __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rax
  _QWORD *v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rsi
  _QWORD *v63; // rcx
  char *v64; // rcx
  _QWORD *v65; // rdi
  _QWORD *v66; // rbx
  int v67; // eax
  int v68; // r9d
  __int64 *v69; // rdi
  unsigned int v70; // r8d
  __int64 *v71; // rcx
  _QWORD *v72; // [rsp+0h] [rbp-90h]
  char *v73; // [rsp+8h] [rbp-88h]
  _QWORD *v74; // [rsp+8h] [rbp-88h]
  int v75; // [rsp+14h] [rbp-7Ch]
  int v76; // [rsp+14h] [rbp-7Ch]
  signed __int64 v77; // [rsp+18h] [rbp-78h]
  __int64 v78; // [rsp+18h] [rbp-78h]
  __int64 v79; // [rsp+20h] [rbp-70h]
  _QWORD *v80; // [rsp+20h] [rbp-70h]
  unsigned __int64 v81; // [rsp+28h] [rbp-68h]
  char *v82; // [rsp+28h] [rbp-68h]
  unsigned __int64 v83; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v84; // [rsp+40h] [rbp-50h] BYREF
  _BYTE *v85; // [rsp+48h] [rbp-48h]
  _BYTE *v86; // [rsp+50h] [rbp-40h]

  v12 = (__int64)a2;
  v13 = a1;
  v14 = *a2;
  if ( *(_BYTE *)(*a2 + 8LL) != 13 )
  {
    v15 = *(_DWORD *)(a1 + 144);
    v16 = *(_QWORD *)(a1 + 128);
    if ( v15 )
    {
      v17 = (v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v12 == *v18 )
      {
LABEL_4:
        v20 = v18[1];
        v21 = 0;
        v22 = (v20 >> 1) & 3;
        if ( v22 == 3 )
          return v21;
        v23 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (unsigned int)(v22 - 1) > 1 )
          v23 = sub_1599EF0((__int64 **)v14);
        goto LABEL_7;
      }
      v67 = 1;
      while ( v19 != -8 )
      {
        v68 = v67 + 1;
        v17 = (v15 - 1) & (v67 + v17);
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v12 == *v18 )
          goto LABEL_4;
        v67 = v68;
      }
    }
    v18 = (__int64 *)(v16 + 16LL * v15);
    goto LABEL_4;
  }
  v25 = *(_DWORD *)(v14 + 12);
  if ( !v25 )
  {
    v82 = 0;
    v39 = 0;
LABEL_33:
    v85 = 0;
    v86 = 0;
    v42 = *(_BYTE *)(v14 + 8) == 13;
    v84 = 0;
    if ( !v42 )
      BUG();
    v43 = *(unsigned int *)(v14 + 12);
    if ( !(_DWORD)v43 )
    {
      v23 = sub_159F090((__int64 **)v14, 0, 0, a12);
LABEL_46:
      if ( v39 )
      {
LABEL_47:
        j_j___libc_free_0(v39, v82 - v39);
        if ( *(_BYTE *)(v12 + 16) != 78 )
          goto LABEL_8;
LABEL_48:
        if ( (*(_WORD *)(v12 + 18) & 3) == 2 )
        {
          LOBYTE(v50) = sub_15F33D0(v12);
          v21 = v50;
          if ( !(_BYTE)v50 )
          {
            v51 = *(_QWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) - 24);
            if ( *(_BYTE *)(v51 + 16) )
              return v21;
            v52 = *(__int64 **)(v13 + 488);
            if ( *(__int64 **)(v13 + 496) == v52 )
            {
              v69 = &v52[*(unsigned int *)(v13 + 508)];
              v70 = *(_DWORD *)(v13 + 508);
              if ( v52 != v69 )
              {
                v71 = 0;
                while ( *v52 != v51 )
                {
                  if ( *v52 == -2 )
                    v71 = v52;
                  if ( v69 == ++v52 )
                  {
                    if ( !v71 )
                      goto LABEL_108;
                    *v71 = v51;
                    --*(_DWORD *)(v13 + 512);
                    ++*(_QWORD *)(v13 + 480);
                    return v21;
                  }
                }
                return v21;
              }
LABEL_108:
              if ( v70 < *(_DWORD *)(v13 + 504) )
              {
                *(_DWORD *)(v13 + 508) = v70 + 1;
                *v69 = v51;
                ++*(_QWORD *)(v13 + 480);
                return v21;
              }
            }
            sub_16CCBA0(v13 + 480, v51);
            return v21;
          }
        }
LABEL_8:
        v21 = 1;
        sub_164D160(v12, v23, a3, a4, a5, a6, a7, a8, a9, a10);
        return v21;
      }
LABEL_7:
      if ( *(_BYTE *)(v12 + 16) != 78 )
        goto LABEL_8;
      goto LABEL_48;
    }
    v79 = v13;
    v44 = 8 * v43;
    v45 = 0;
    v46 = v14;
    while ( 1 )
    {
      while ( 1 )
      {
        v49 = (*(__int64 *)&v39[v45] >> 1) & 3;
        if ( v49 != 1 && v49 != 2 )
          break;
        v47 = *(_QWORD *)&v39[v45] & 0xFFFFFFFFFFFFFFF8LL;
        v48 = v85;
        v83 = v47;
        if ( v85 != v86 )
          goto LABEL_38;
LABEL_43:
        v45 += 8;
        sub_12F5DA0((__int64)&v84, v48, &v83);
        if ( v44 == v45 )
          goto LABEL_44;
      }
      v47 = sub_1599EF0(*(__int64 ***)(*(_QWORD *)(v46 + 16) + v45));
      v48 = v85;
      v83 = v47;
      if ( v85 == v86 )
        goto LABEL_43;
LABEL_38:
      if ( v48 )
      {
        *(_QWORD *)v48 = v47;
        v48 = v85;
      }
      v45 += 8;
      v85 = v48 + 8;
      if ( v44 == v45 )
      {
LABEL_44:
        v13 = v79;
        v23 = sub_159F090((__int64 **)v46, v84, (v85 - (_BYTE *)v84) >> 3, a12);
        if ( v84 )
        {
          j_j___libc_free_0(v84, v86 - (_BYTE *)v84);
          goto LABEL_46;
        }
        goto LABEL_47;
      }
    }
  }
  v26 = 0;
  v27 = 0;
  v28 = a2;
  v29 = 0;
  v81 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
  v30 = 0;
  do
  {
    v31 = *(unsigned int *)(v13 + 208);
    v32 = *(_QWORD *)(v13 + 192);
    if ( (_DWORD)v31 )
    {
      v33 = 1;
      v34 = ((((unsigned int)(37 * v27) | v81) - 1 - ((unsigned __int64)(unsigned int)(37 * v27) << 32)) >> 22)
          ^ (((unsigned int)(37 * v27) | v81) - 1 - ((unsigned __int64)(unsigned int)(37 * v27) << 32));
      v35 = ((9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13)))) >> 15)
          ^ (9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13))));
      for ( i = (v31 - 1) & (((v35 - 1 - (v35 << 27)) >> 31) ^ (v35 - 1 - ((_DWORD)v35 << 27))); ; i = (v31 - 1) & v38 )
      {
        v37 = v32 + 24LL * i;
        if ( v28 == *(_QWORD **)v37 && *(_DWORD *)(v37 + 8) == v27 )
          break;
        if ( *(_QWORD *)v37 == -8 && *(_DWORD *)(v37 + 8) == -1 )
          goto LABEL_19;
        v38 = v33 + i;
        ++v33;
      }
      if ( v29 == v26 )
      {
LABEL_55:
        v53 = v29 - v30;
        v54 = (v29 - v30) >> 3;
        if ( v54 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v55 = 1;
        if ( v54 )
          v55 = (v29 - v30) >> 3;
        v56 = __CFADD__(v55, v54);
        v57 = v55 + v54;
        if ( v56 )
        {
          v58 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v57 )
          {
            v62 = 8;
            v61 = 0;
            v60 = 0;
            goto LABEL_64;
          }
          if ( v57 > 0xFFFFFFFFFFFFFFFLL )
            v57 = 0xFFFFFFFFFFFFFFFLL;
          v58 = 8 * v57;
        }
        v72 = v28;
        v73 = v30;
        v75 = v25;
        v77 = v29 - v30;
        v59 = sub_22077B0(v58);
        v53 = v77;
        v60 = (_QWORD *)v59;
        v25 = v75;
        v30 = v73;
        v28 = v72;
        v61 = v58 + v59;
        v62 = (__int64)(v60 + 1);
LABEL_64:
        v63 = (_QWORD *)((char *)v60 + v53);
        if ( v63 )
          *v63 = *(_QWORD *)(v37 + 16);
        if ( v26 == v30 )
        {
          v26 = (char *)v62;
        }
        else
        {
          v64 = v30;
          v65 = (_QWORD *)((char *)v60 + v26 - v30);
          v66 = v60;
          do
          {
            if ( v66 )
              *v66 = *(_QWORD *)v64;
            ++v66;
            v64 += 8;
          }
          while ( v66 != v65 );
          v26 = (char *)(v66 + 1);
        }
        if ( v30 )
        {
          v74 = v28;
          v76 = v25;
          v78 = v61;
          v80 = v60;
          j_j___libc_free_0(v30, v29 - v30);
          v28 = v74;
          v25 = v76;
          v61 = v78;
          v60 = v80;
        }
        v29 = (char *)v61;
        v30 = (char *)v60;
        goto LABEL_23;
      }
    }
    else
    {
LABEL_19:
      v37 = v32 + 24 * v31;
      if ( v29 == v26 )
        goto LABEL_55;
    }
    if ( v26 )
      *(_QWORD *)v26 = *(_QWORD *)(v37 + 16);
    v26 += 8;
LABEL_23:
    ++v27;
  }
  while ( v25 != v27 );
  v82 = v29;
  v39 = v30;
  v12 = (__int64)v28;
  a12 = (v26 - v30) >> 5;
  v40 = (v26 - v30) >> 3;
  if ( a12 <= 0 )
  {
    v41 = v30;
LABEL_83:
    if ( v40 != 2 )
    {
      if ( v40 != 3 )
      {
        if ( v40 != 1 )
          goto LABEL_32;
        goto LABEL_86;
      }
      if ( (((unsigned __int8)*(_QWORD *)v41 ^ 6) & 6) == 0 )
        goto LABEL_31;
      v41 += 8;
    }
    if ( (((unsigned __int8)*(_QWORD *)v41 ^ 6) & 6) == 0 )
      goto LABEL_31;
    v41 += 8;
LABEL_86:
    if ( (((unsigned __int8)*(_QWORD *)v41 ^ 6) & 6) == 0 )
      goto LABEL_31;
    goto LABEL_32;
  }
  v41 = v30;
  a12 = (__int64)&v30[32 * a12];
  while ( 1 )
  {
    if ( (((unsigned __int8)*(_QWORD *)v41 ^ 6) & 6) == 0 )
      goto LABEL_31;
    if ( (((unsigned __int8)*((_QWORD *)v41 + 1) ^ 6) & 6) == 0 )
    {
      v41 += 8;
      goto LABEL_31;
    }
    if ( (((unsigned __int8)*((_QWORD *)v41 + 2) ^ 6) & 6) == 0 )
    {
      v41 += 16;
      goto LABEL_31;
    }
    if ( (((unsigned __int8)*((_QWORD *)v41 + 3) ^ 6) & 6) == 0 )
      break;
    v41 += 32;
    if ( (char *)a12 == v41 )
    {
      v40 = (v26 - v41) >> 3;
      goto LABEL_83;
    }
  }
  v41 += 24;
LABEL_31:
  if ( v26 == v41 )
  {
LABEL_32:
    v14 = *v28;
    goto LABEL_33;
  }
  if ( v30 )
    j_j___libc_free_0(v30, v82 - v30);
  return 0;
}
