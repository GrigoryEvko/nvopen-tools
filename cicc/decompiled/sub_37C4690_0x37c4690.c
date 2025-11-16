// Function: sub_37C4690
// Address: 0x37c4690
//
char *__fastcall sub_37C4690(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r10
  __int64 v7; // rdx
  _QWORD *v8; // r14
  unsigned int v9; // r12d
  __int64 v10; // r13
  char v11; // si
  __int64 v12; // rdi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r11
  int v17; // edx
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  unsigned __int64 *v23; // rax
  unsigned __int64 *i; // rsi
  int v25; // eax
  __int64 v26; // r11
  __int64 v27; // r12
  __int64 v28; // rax
  int v29; // r13d
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rdi
  __int64 v35; // rcx
  unsigned __int64 *v36; // r14
  char *v37; // r12
  unsigned int v38; // ebx
  __int64 v39; // r13
  int v40; // r14d
  __int64 v41; // rax
  char *p_src; // r15
  unsigned __int64 *v43; // rax
  char *v44; // r8
  unsigned int *v45; // rbx
  __int64 v46; // r9
  char *v47; // r14
  unsigned int v48; // r13d
  unsigned int v49; // eax
  char *v50; // rax
  __int64 v51; // rdi
  int v52; // edx
  unsigned __int64 v53; // r11
  char *v54; // rsi
  unsigned int v55; // edx
  unsigned __int64 *v56; // rbx
  unsigned __int64 *v57; // r12
  int v59; // eax
  int v60; // r15d
  __int64 v62; // [rsp+18h] [rbp-1E8h]
  __int64 v63; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v64; // [rsp+20h] [rbp-1E0h]
  __int64 v65; // [rsp+20h] [rbp-1E0h]
  __int64 v66; // [rsp+20h] [rbp-1E0h]
  __int64 v67; // [rsp+28h] [rbp-1D8h]
  __int64 v68; // [rsp+30h] [rbp-1D0h]
  __int64 v70; // [rsp+38h] [rbp-1C8h]
  unsigned int dest; // [rsp+40h] [rbp-1C0h]
  __int64 v72; // [rsp+48h] [rbp-1B8h]
  char *v73; // [rsp+48h] [rbp-1B8h]
  char *v74; // [rsp+48h] [rbp-1B8h]
  __int64 v75; // [rsp+48h] [rbp-1B8h]
  __int64 v76; // [rsp+50h] [rbp-1B0h]
  __int64 v77; // [rsp+50h] [rbp-1B0h]
  char *v78; // [rsp+50h] [rbp-1B0h]
  char *v79; // [rsp+50h] [rbp-1B0h]
  unsigned int v80; // [rsp+58h] [rbp-1A8h]
  __int64 v81; // [rsp+58h] [rbp-1A8h]
  __int64 v82; // [rsp+58h] [rbp-1A8h]
  __int64 v83; // [rsp+58h] [rbp-1A8h]
  __int64 v84; // [rsp+60h] [rbp-1A0h]
  __int64 v85; // [rsp+60h] [rbp-1A0h]
  __int64 *v86; // [rsp+68h] [rbp-198h]
  int v87; // [rsp+68h] [rbp-198h]
  char *v88; // [rsp+80h] [rbp-180h] BYREF
  __int64 v89; // [rsp+88h] [rbp-178h]
  _BYTE v90[16]; // [rsp+90h] [rbp-170h] BYREF
  char *v91; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v92; // [rsp+A8h] [rbp-158h]
  char src; // [rsp+B0h] [rbp-150h] BYREF
  unsigned __int64 *v94; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v95; // [rsp+C8h] [rbp-138h]
  _BYTE v96[304]; // [rsp+D0h] [rbp-130h] BYREF

  v6 = a3;
  v94 = (unsigned __int64 *)v96;
  v95 = 0x800000000LL;
  v7 = *(unsigned int *)(a6 + 8);
  dest = *(_DWORD *)(a1[51] + 40LL);
  v72 = *(_QWORD *)a6 + 8 * v7;
  if ( *(_QWORD *)a6 != v72 )
  {
    v86 = *(__int64 **)a6;
    v8 = (_QWORD *)a5;
    v9 = 0;
    v67 = a2;
    a6 = 0xFFFFFF0000000000LL;
    a5 = 0xFFFFFFFFFFLL;
    v68 = 4LL * a2;
    v80 = 8;
    v76 = a4;
    while ( 1 )
    {
      v10 = *v86;
      v11 = *(_BYTE *)(v76 + 8) & 1;
      if ( v11 )
      {
        v12 = v76 + 16;
        v13 = 15;
      }
      else
      {
        v31 = *(unsigned int *)(v76 + 24);
        v12 = *(_QWORD *)(v76 + 16);
        if ( !(_DWORD)v31 )
          goto LABEL_44;
        v13 = v31 - 1;
      }
      v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_6;
      v59 = 1;
      while ( v16 != -4096 )
      {
        v60 = v59 + 1;
        v14 = v13 & (v59 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v10 == *v15 )
          goto LABEL_6;
        v59 = v60;
      }
      if ( v11 )
      {
        v35 = 256;
        goto LABEL_45;
      }
      v31 = *(unsigned int *)(v76 + 24);
LABEL_44:
      v35 = 16 * v31;
LABEL_45:
      v15 = (__int64 *)(v12 + v35);
LABEL_6:
      a4 = v15[1];
      if ( *(_DWORD *)(a4 + 32) )
      {
        v17 = *(_DWORD *)(a4 + 4 * v67);
        v18 = *(_DWORD *)(a4 + v68) >> 1;
      }
      else
      {
        LOBYTE(v17) = dword_5051178[0];
        v18 = dword_5051178[0] >> 1;
      }
      v19 = (2 * v18) | v17 & 1;
      if ( v19 == dword_5051178[0] )
      {
        v7 = 0;
        v20 = unk_5051170;
      }
      else
      {
        v7 = v19 & 1;
        if ( (_DWORD)v7 )
          v20 = *(_QWORD *)(a1[273] + 40 * v18);
        else
          v20 = *(_QWORD *)(a1[271] + 8 * v18);
      }
      v21 = v9;
      v22 = v9 + 1LL;
      if ( v80 < v22 )
      {
        v62 = v6;
        v64 = v7;
        v81 = a4;
        sub_37C4590((__int64)&v94, v22, v7, a4, 0xFFFFFFFFFFLL, 0xFFFFFF0000000000LL);
        v21 = (unsigned int)v95;
        v6 = v62;
        a5 = 0xFFFFFFFFFFLL;
        v7 = v64;
        a4 = v81;
        a6 = 0xFFFFFF0000000000LL;
        v22 = v9 + 1LL;
      }
      v23 = &v94[4 * v21];
      for ( i = &v94[4 * v22]; i != v23; v23 += 4 )
      {
        if ( v23 )
        {
          *((_DWORD *)v23 + 2) = 0;
          *v23 = (unsigned __int64)(v23 + 2);
          *((_DWORD *)v23 + 3) = 4;
        }
      }
      v25 = *(_DWORD *)(a4 + 56);
      LODWORD(v95) = v9 + 1;
      if ( v25 == 1 || v25 == 2 && *(_DWORD *)(a4 + 36) != *(_DWORD *)(v6 + 24) && ((_BYTE)v7 || unk_5051170 != v20) )
      {
        if ( dest )
        {
          v26 = dest;
          v27 = 0;
          v28 = v10;
          do
          {
            while ( 1 )
            {
              a4 = *(int *)(v28 + 24);
              v29 = v27;
              v7 = **(_QWORD **)(*v8 + 8 * a4);
              if ( *(_QWORD *)(v7 + 8 * v27) == v20 )
                break;
              if ( ++v27 == v26 )
                goto LABEL_29;
            }
            v30 = (__int64)&v94[4 * (unsigned int)v95 - 4];
            a4 = *(unsigned int *)(v30 + 8);
            if ( a4 + 1 > (unsigned __int64)*(unsigned int *)(v30 + 12) )
            {
              v63 = v6;
              v66 = v26;
              v83 = v28;
              v85 = (__int64)&v94[4 * (unsigned int)v95 - 4];
              sub_C8D5F0(v30, (const void *)(v30 + 16), a4 + 1, 4u, 0xFFFFFFFFFFLL, 0xFFFFFF0000000000LL);
              v30 = v85;
              v6 = v63;
              a5 = 0xFFFFFFFFFFLL;
              v26 = v66;
              v28 = v83;
              a6 = 0xFFFFFF0000000000LL;
              a4 = *(unsigned int *)(v85 + 8);
            }
            v7 = *(_QWORD *)v30;
            ++v27;
            *(_DWORD *)(*(_QWORD *)v30 + 4 * a4) = v29;
            ++*(_DWORD *)(v30 + 8);
          }
          while ( v27 != v26 );
        }
      }
      else
      {
        v32 = dest;
        v33 = 0;
        if ( dest )
        {
          do
          {
            a4 = *(int *)(v10 + 24);
            v7 = **(_QWORD **)(*v8 + 8 * a4);
            if ( *(_QWORD *)(v7 + 8 * v33) == ((v33 << 40) | *(_DWORD *)(v6 + 24) & 0xFFFFFLL) )
            {
              v34 = (__int64)&v94[4 * (unsigned int)v95 - 4];
              v7 = *(unsigned int *)(v34 + 8);
              if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 12) )
              {
                v65 = v6;
                v82 = v32;
                v84 = (__int64)&v94[4 * (unsigned int)v95 - 4];
                sub_C8D5F0(v34, (const void *)(v34 + 16), v7 + 1, 4u, 0xFFFFFFFFFFLL, 0xFFFFFF0000000000LL);
                v34 = v84;
                v6 = v65;
                a5 = 0xFFFFFFFFFFLL;
                v32 = v82;
                a6 = 0xFFFFFF0000000000LL;
                v7 = *(unsigned int *)(v84 + 8);
              }
              a4 = *(_QWORD *)v34;
              *(_DWORD *)(*(_QWORD *)v34 + 4 * v7) = v33;
              ++*(_DWORD *)(v34 + 8);
            }
            ++v33;
          }
          while ( v32 != v33 );
        }
      }
LABEL_29:
      if ( (__int64 *)v72 == ++v86 )
      {
        v36 = v94;
        goto LABEL_47;
      }
      v9 = v95;
      v80 = HIDWORD(v95);
    }
  }
  v36 = (unsigned __int64 *)v96;
LABEL_47:
  v37 = v90;
  v89 = 0x400000000LL;
  v38 = *((_DWORD *)v36 + 2);
  v88 = v90;
  if ( v38 )
  {
    if ( v36 != (unsigned __int64 *)&v88 )
    {
      v39 = v38;
      v37 = v90;
      v7 = 4LL * v38;
      if ( v38 <= 4
        || (sub_C8D5F0((__int64)&v88, v90, v38, 4u, a5, a6), v37 = v88, (v7 = 4LL * *((unsigned int *)v36 + 2)) != 0) )
      {
        memcpy(v37, (const void *)*v36, v7);
        v37 = v88;
      }
      LODWORD(v89) = v38;
      if ( (unsigned int)v95 <= 1 )
      {
LABEL_77:
        v55 = *(_DWORD *)v37;
        LOBYTE(v92) = 1;
        LODWORD(v91) = *(_DWORD *)(a3 + 24) & 0xFFFFF;
        HIDWORD(v91) = v55 << 8;
        goto LABEL_78;
      }
      goto LABEL_49;
    }
    v37 = v90;
  }
  v39 = 0;
  if ( (unsigned int)v95 <= 1 )
    goto LABEL_88;
LABEL_49:
  v40 = 1;
  v41 = 1;
  p_src = &src;
  while ( 1 )
  {
    v43 = &v94[4 * v41];
    v91 = p_src;
    v44 = &v37[4 * v39];
    v92 = 0x400000000LL;
    v45 = (unsigned int *)*v43;
    v46 = *v43 + 4LL * *((unsigned int *)v43 + 2);
    if ( v44 != v37 && v45 != (unsigned int *)v46 )
    {
      v87 = v40;
      v47 = &v37[4 * v39];
      v44 = p_src;
      while ( 1 )
      {
        v48 = *(_DWORD *)v37;
        v49 = *v45;
        if ( *(_DWORD *)v37 >= *v45 )
        {
          ++v45;
          if ( v48 > v49 )
          {
            if ( v47 == v37 )
              goto LABEL_59;
            goto LABEL_55;
          }
          v50 = v91;
          v51 = 4LL * (unsigned int)v92;
          v52 = v92;
          v53 = (unsigned int)v92 + 1LL;
          a4 = HIDWORD(v92);
          v54 = &v91[v51];
          if ( &v91[v51] == p_src )
          {
            if ( v53 > HIDWORD(v92) )
            {
              v75 = v46;
              v79 = v44;
              sub_C8D5F0((__int64)&v91, v44, (unsigned int)v92 + 1LL, 4u, (__int64)v44, v46);
              v46 = v75;
              v44 = v79;
              p_src = &v91[4 * (unsigned int)v92];
            }
            *(_DWORD *)p_src = v48;
            v7 = (__int64)v91;
            LODWORD(v92) = v92 + 1;
            p_src = &v91[4 * (unsigned int)v92 - 4];
          }
          else
          {
            if ( v53 > HIDWORD(v92) )
            {
              v70 = v46;
              v74 = v91;
              v78 = v44;
              sub_C8D5F0((__int64)&v91, v44, (unsigned int)v92 + 1LL, 4u, (__int64)v44, v46);
              v46 = v70;
              v52 = v92;
              v51 = 4LL * (unsigned int)v92;
              v44 = v78;
              p_src = &v91[p_src - v74];
              v54 = &v91[v51];
              v50 = v91;
            }
            a4 = (__int64)&v50[v51 - 4];
            if ( v54 )
            {
              *(_DWORD *)v54 = *(_DWORD *)a4;
              v50 = v91;
              v52 = v92;
              v51 = 4LL * (unsigned int)v92;
              a4 = (__int64)&v91[v51 - 4];
            }
            if ( p_src != (char *)a4 )
            {
              v73 = v44;
              v77 = v46;
              memmove(&v50[v51 - (a4 - (_QWORD)p_src)], p_src, a4 - (_QWORD)p_src);
              v52 = v92;
              v44 = v73;
              v46 = v77;
            }
            v7 = (unsigned int)(v52 + 1);
            LODWORD(v92) = v7;
            *(_DWORD *)p_src = v48;
          }
          p_src += 4;
          v37 += 4;
        }
        else
        {
          v37 += 4;
        }
        if ( v47 == v37 )
        {
LABEL_59:
          v40 = v87;
          p_src = v44;
          break;
        }
LABEL_55:
        if ( (unsigned int *)v46 == v45 )
          goto LABEL_59;
      }
    }
    sub_37B6D10((__int64)&v88, &v91, v7, a4, (__int64)v44, v46);
    if ( v91 != p_src )
      _libc_free((unsigned __int64)v91);
    v41 = (unsigned int)(v40 + 1);
    v37 = v88;
    v40 = v41;
    if ( (unsigned int)v41 >= (unsigned int)v95 )
      break;
    v39 = (unsigned int)v89;
  }
  if ( (_DWORD)v89 )
    goto LABEL_77;
LABEL_88:
  LOBYTE(v92) = 0;
LABEL_78:
  if ( v37 != v90 )
    _libc_free((unsigned __int64)v37);
  v56 = v94;
  v57 = &v94[4 * (unsigned int)v95];
  if ( v94 != v57 )
  {
    do
    {
      v57 -= 4;
      if ( (unsigned __int64 *)*v57 != v57 + 2 )
        _libc_free(*v57);
    }
    while ( v56 != v57 );
    v57 = v94;
  }
  if ( v57 != (unsigned __int64 *)v96 )
    _libc_free((unsigned __int64)v57);
  return v91;
}
