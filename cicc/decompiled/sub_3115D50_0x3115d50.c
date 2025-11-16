// Function: sub_3115D50
// Address: 0x3115d50
//
__int64 __fastcall sub_3115D50(
        unsigned __int64 **a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 *v6; // rdi
  _QWORD *v7; // r12
  __int64 i; // r14
  unsigned int *v9; // r15
  char *v10; // rdi
  unsigned __int64 v11; // rax
  unsigned int *v12; // r13
  unsigned int v13; // ecx
  unsigned int v14; // edx
  unsigned int *v15; // rax
  unsigned int *v16; // rsi
  unsigned int *v18; // rsi
  __int64 **v19; // rdx
  __int64 v20; // rax
  __int64 **v21; // r13
  __int64 *v22; // rax
  __int64 **v23; // r15
  _QWORD *v24; // r12
  char *v25; // rbx
  char *v26; // r11
  char *v27; // r15
  int v28; // r10d
  __int64 **v29; // rdx
  unsigned int v30; // edi
  __int64 **v31; // rax
  __int64 *v32; // rcx
  _DWORD *v33; // r14
  __int64 *v34; // r13
  unsigned int v35; // ecx
  int v36; // eax
  __int64 *v37; // rdi
  signed __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rcx
  bool v41; // cf
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // r13
  __int64 v44; // rax
  char *v45; // rcx
  unsigned __int64 v46; // r13
  signed __int64 v47; // rcx
  unsigned __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rax
  _BYTE *v54; // rdi
  _BYTE *v55; // r8
  size_t v56; // rdx
  char *v57; // r14
  __int64 **v58; // r8
  unsigned int v59; // r14d
  int v60; // r10d
  __int64 *v61; // rsi
  char *v62; // rax
  __int64 v63; // rax
  char *v64; // rcx
  char *v65; // rax
  unsigned __int64 v66; // rdi
  int v67; // r14d
  __int64 **v68; // r9
  char *src; // [rsp+18h] [rbp-C8h]
  size_t n; // [rsp+20h] [rbp-C0h]
  size_t nb; // [rsp+20h] [rbp-C0h]
  size_t nc; // [rsp+20h] [rbp-C0h]
  size_t na; // [rsp+20h] [rbp-C0h]
  size_t nd; // [rsp+20h] [rbp-C0h]
  size_t ne; // [rsp+20h] [rbp-C0h]
  __int64 **v76; // [rsp+28h] [rbp-B8h]
  signed __int64 v77; // [rsp+28h] [rbp-B8h]
  __int64 **v79; // [rsp+38h] [rbp-A8h]
  __int64 v80; // [rsp+40h] [rbp-A0h]
  char *v81; // [rsp+40h] [rbp-A0h]
  unsigned int *v82; // [rsp+48h] [rbp-98h]
  int v83; // [rsp+48h] [rbp-98h]
  char *v84; // [rsp+48h] [rbp-98h]
  char *v85; // [rsp+48h] [rbp-98h]
  char *v86; // [rsp+48h] [rbp-98h]
  char *v87; // [rsp+48h] [rbp-98h]
  char *v88; // [rsp+48h] [rbp-98h]
  char *v89; // [rsp+48h] [rbp-98h]
  __int64 v90; // [rsp+50h] [rbp-90h] BYREF
  __int64 **v91; // [rsp+58h] [rbp-88h]
  __int64 v92; // [rsp+60h] [rbp-80h]
  unsigned int v93; // [rsp+68h] [rbp-78h]
  _QWORD v94[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 (__fastcall *v95)(_QWORD *, _QWORD *, int); // [rsp+80h] [rbp-60h]
  __int64 *(__fastcall *v96)(__int64 *, __int64 *); // [rsp+88h] [rbp-58h]
  unsigned int *v97[2]; // [rsp+90h] [rbp-50h] BYREF
  void (__fastcall *v98)(unsigned int **, unsigned int **, __int64); // [rsp+A0h] [rbp-40h]

  v94[0] = &v90;
  v6 = *a1;
  v96 = sub_3115B10;
  v95 = sub_3114B70;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v98 = 0;
  sub_31144C0(v6, (__int64)v94, (__int64)v97, 1, a5, a6);
  if ( v95 )
    v95(v94, v94, 3);
  if ( v98 )
    v98(v97, v97, 3);
  v7 = a2 + 1;
  if ( (_DWORD)v92 )
  {
    v19 = v91;
    v20 = 2LL * v93;
    v21 = &v91[v20];
    if ( v91 != &v91[v20] )
    {
      while ( 1 )
      {
        v22 = *v19;
        v23 = v19;
        if ( *v19 != (__int64 *)-8192LL && v22 != (__int64 *)-4096LL )
          break;
        v19 += 2;
        if ( v21 == v19 )
          goto LABEL_23;
      }
      if ( v21 != v19 )
      {
        v79 = v21;
        while ( 1 )
        {
          v83 = 0;
          LODWORD(v94[0]) = *((_DWORD *)v23 + 2);
          v80 = *v22;
          if ( *((_BYTE *)v22 + 12) )
            v83 = *((_DWORD *)v22 + 2);
          v24 = (_QWORD *)v22[4];
          v25 = 0;
          v26 = 0;
          if ( v24 )
          {
            v76 = v23;
            v27 = 0;
            while ( 1 )
            {
              v34 = (__int64 *)v24[2];
              if ( !v93 )
                break;
              v28 = 1;
              v29 = 0;
              v30 = (v93 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
              v31 = &v91[2 * v30];
              v32 = *v31;
              if ( v34 == *v31 )
              {
LABEL_31:
                v33 = v31 + 1;
                if ( v25 == v27 )
                  goto LABEL_42;
LABEL_32:
                if ( v25 )
                  *(_DWORD *)v25 = *v33;
                v24 = (_QWORD *)*v24;
                v25 += 4;
                if ( !v24 )
                {
LABEL_56:
                  v23 = v76;
                  v47 = v25 - v26;
                  v48 = v25 - v26;
                  goto LABEL_57;
                }
              }
              else
              {
                while ( v32 != (__int64 *)-4096LL )
                {
                  if ( !v29 && v32 == (__int64 *)-8192LL )
                    v29 = v31;
                  v30 = (v93 - 1) & (v28 + v30);
                  v31 = &v91[2 * v30];
                  v32 = *v31;
                  if ( v34 == *v31 )
                    goto LABEL_31;
                  ++v28;
                }
                if ( !v29 )
                  v29 = v31;
                ++v90;
                v36 = v92 + 1;
                if ( 4 * ((int)v92 + 1) >= 3 * v93 )
                  goto LABEL_37;
                if ( v93 - HIDWORD(v92) - v36 <= v93 >> 3 )
                {
                  na = (size_t)v26;
                  sub_3115930((__int64)&v90, v93);
                  if ( !v93 )
                  {
LABEL_131:
                    LODWORD(v92) = v92 + 1;
                    BUG();
                  }
                  v58 = 0;
                  v59 = (v93 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                  v26 = (char *)na;
                  v60 = 1;
                  v36 = v92 + 1;
                  v29 = &v91[2 * v59];
                  v61 = *v29;
                  if ( v34 != *v29 )
                  {
                    while ( v61 != (__int64 *)-4096LL )
                    {
                      if ( !v58 && v61 == (__int64 *)-8192LL )
                        v58 = v29;
                      v59 = (v93 - 1) & (v60 + v59);
                      v29 = &v91[2 * v59];
                      v61 = *v29;
                      if ( v34 == *v29 )
                        goto LABEL_39;
                      ++v60;
                    }
                    if ( v58 )
                      v29 = v58;
                  }
                }
LABEL_39:
                LODWORD(v92) = v36;
                if ( *v29 != (__int64 *)-4096LL )
                  --HIDWORD(v92);
                *v29 = v34;
                v33 = v29 + 1;
                *((_DWORD *)v29 + 2) = 0;
                if ( v25 != v27 )
                  goto LABEL_32;
LABEL_42:
                v38 = v25 - v26;
                v39 = (v25 - v26) >> 2;
                if ( v39 == 0x1FFFFFFFFFFFFFFFLL )
                  sub_4262D8((__int64)"vector::_M_realloc_insert");
                v40 = 1;
                if ( v39 )
                  v40 = v38 >> 2;
                v41 = __CFADD__(v40, v39);
                v42 = v40 + v39;
                if ( v41 )
                {
                  v43 = 0x7FFFFFFFFFFFFFFCLL;
LABEL_50:
                  src = v26;
                  nb = v25 - v26;
                  v44 = sub_22077B0(v43);
                  v38 = nb;
                  v26 = src;
                  v45 = (char *)v44;
                  v46 = v44 + v43;
                  goto LABEL_51;
                }
                if ( v42 )
                {
                  if ( v42 > 0x1FFFFFFFFFFFFFFFLL )
                    v42 = 0x1FFFFFFFFFFFFFFFLL;
                  v43 = 4 * v42;
                  goto LABEL_50;
                }
                v46 = 0;
                v45 = 0;
LABEL_51:
                if ( &v45[v38] )
                  *(_DWORD *)&v45[v38] = *v33;
                v25 = &v45[v38 + 4];
                if ( v38 > 0 )
                {
                  nd = (size_t)v26;
                  v62 = (char *)memmove(v45, v26, v38);
                  v26 = (char *)nd;
                  v45 = v62;
LABEL_96:
                  ne = (size_t)v45;
                  j_j___libc_free_0((unsigned __int64)v26);
                  v45 = (char *)ne;
                  goto LABEL_55;
                }
                if ( v26 )
                  goto LABEL_96;
LABEL_55:
                v24 = (_QWORD *)*v24;
                v27 = (char *)v46;
                v26 = v45;
                if ( !v24 )
                  goto LABEL_56;
              }
            }
            ++v90;
LABEL_37:
            n = (size_t)v26;
            sub_3115930((__int64)&v90, 2 * v93);
            if ( !v93 )
              goto LABEL_131;
            v26 = (char *)n;
            v35 = (v93 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v36 = v92 + 1;
            v29 = &v91[2 * v35];
            v37 = *v29;
            if ( v34 != *v29 )
            {
              v67 = 1;
              v68 = 0;
              while ( v37 != (__int64 *)-4096LL )
              {
                if ( v37 == (__int64 *)-8192LL && !v68 )
                  v68 = v29;
                v35 = (v93 - 1) & (v67 + v35);
                v29 = &v91[2 * v35];
                v37 = *v29;
                if ( v34 == *v29 )
                  goto LABEL_39;
                ++v67;
              }
              if ( v68 )
                v29 = v68;
            }
            goto LABEL_39;
          }
          v48 = 0;
          v47 = 0;
LABEL_57:
          v7 = a2 + 1;
          v49 = a2[2];
          if ( !v49 )
            break;
          v50 = (__int64)(a2 + 1);
          do
          {
            while ( 1 )
            {
              v51 = *(_QWORD *)(v49 + 16);
              v52 = *(_QWORD *)(v49 + 24);
              if ( *(_DWORD *)(v49 + 32) >= LODWORD(v94[0]) )
                break;
              v49 = *(_QWORD *)(v49 + 24);
              if ( !v52 )
                goto LABEL_62;
            }
            v50 = v49;
            v49 = *(_QWORD *)(v49 + 16);
          }
          while ( v51 );
LABEL_62:
          if ( (_QWORD *)v50 == v7 || LODWORD(v94[0]) < *(_DWORD *)(v50 + 32) )
            goto LABEL_64;
LABEL_65:
          v54 = *(_BYTE **)(v50 + 56);
          *(_QWORD *)(v50 + 40) = v80;
          *(_DWORD *)(v50 + 48) = v83;
          if ( v48 > *(_QWORD *)(v50 + 72) - (_QWORD)v54 )
          {
            if ( v47 )
            {
              if ( v48 > 0x7FFFFFFFFFFFFFFCLL )
                sub_4261EA(v54, v51, v52);
              v87 = v26;
              v63 = sub_22077B0(v48);
              v26 = v87;
              v64 = (char *)v63;
            }
            else
            {
              v64 = 0;
            }
            if ( v26 != v25 )
            {
              v88 = v26;
              v65 = (char *)memcpy(v64, v26, v48);
              v26 = v88;
              v64 = v65;
            }
            v66 = *(_QWORD *)(v50 + 56);
            if ( v66 )
            {
              v81 = v26;
              v89 = v64;
              j_j___libc_free_0(v66);
              v26 = v81;
              v64 = v89;
            }
            v57 = &v64[v48];
            *(_QWORD *)(v50 + 56) = v64;
            *(_QWORD *)(v50 + 72) = v57;
          }
          else
          {
            v55 = *(_BYTE **)(v50 + 64);
            v56 = v55 - v54;
            if ( v48 <= v55 - v54 )
            {
              if ( v26 != v25 )
              {
                v84 = v26;
                memmove(v54, v26, v48);
                v54 = *(_BYTE **)(v50 + 56);
                v26 = v84;
              }
LABEL_69:
              v57 = &v54[v48];
              goto LABEL_70;
            }
            if ( v56 )
            {
              v85 = v26;
              memmove(v54, v26, v56);
              v55 = *(_BYTE **)(v50 + 64);
              v54 = *(_BYTE **)(v50 + 56);
              v26 = v85;
              v56 = v55 - v54;
            }
            if ( &v26[v56] == v25 )
              goto LABEL_69;
            v86 = v26;
            memmove(v55, &v26[v56], v25 - &v26[v56]);
            v26 = v86;
            v57 = (char *)(*(_QWORD *)(v50 + 56) + v48);
          }
LABEL_70:
          *(_QWORD *)(v50 + 64) = v57;
          if ( v26 )
            j_j___libc_free_0((unsigned __int64)v26);
          v23 += 2;
          if ( v23 != v79 )
          {
            while ( 1 )
            {
              v22 = *v23;
              if ( *v23 != (__int64 *)-8192LL && v22 != (__int64 *)-4096LL )
                break;
              v23 += 2;
              if ( v79 == v23 )
                goto LABEL_6;
            }
            if ( v79 != v23 )
              continue;
          }
          goto LABEL_6;
        }
        v50 = (__int64)(a2 + 1);
LABEL_64:
        v51 = v50;
        nc = (size_t)v26;
        v77 = v47;
        v97[0] = (unsigned int *)v94;
        v53 = sub_3115870(a2, v50, v97);
        v26 = (char *)nc;
        v47 = v77;
        v50 = v53;
        goto LABEL_65;
      }
LABEL_23:
      v7 = a2 + 1;
    }
  }
LABEL_6:
  for ( i = a2[3]; (_QWORD *)i != v7; i = sub_220EEE0(i) )
  {
    while ( 1 )
    {
      v9 = *(unsigned int **)(i + 64);
      v10 = *(char **)(i + 56);
      if ( v9 != (unsigned int *)v10 )
        break;
LABEL_13:
      i = sub_220EEE0(i);
      if ( (_QWORD *)i == v7 )
        return sub_C7D6A0((__int64)v91, 16LL * v93, 8);
    }
    v82 = *(unsigned int **)(i + 56);
    _BitScanReverse64(&v11, ((char *)v9 - v10) >> 2);
    sub_3114F50(v10, *(char **)(i + 64), 2LL * (int)(63 - (v11 ^ 0x3F)));
    if ( (char *)v9 - v10 > 64 )
    {
      v12 = v82 + 16;
      sub_3114CC0(v82, v82 + 16);
      if ( v9 != v82 + 16 )
      {
        do
        {
          while ( 1 )
          {
            v13 = *v12;
            v14 = *(v12 - 1);
            v15 = v12 - 1;
            if ( *v12 < v14 )
              break;
            v18 = v12++;
            *v18 = v13;
            if ( v9 == v12 )
              goto LABEL_13;
          }
          do
          {
            v15[1] = v14;
            v16 = v15;
            v14 = *--v15;
          }
          while ( v13 < v14 );
          ++v12;
          *v16 = v13;
        }
        while ( v9 != v12 );
      }
      goto LABEL_13;
    }
    sub_3114CC0(v82, v9);
  }
  return sub_C7D6A0((__int64)v91, 16LL * v93, 8);
}
