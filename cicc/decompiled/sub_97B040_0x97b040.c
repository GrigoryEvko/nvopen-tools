// Function: sub_97B040
// Address: 0x97b040
//
__int64 __fastcall sub_97B040(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v7; // rax
  char **v8; // rbx
  char **v9; // r15
  _BYTE *v10; // r8
  char *v12; // r12
  char v13; // dl
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  _BYTE *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r12
  char v21; // cl
  char *v22; // r9
  int v23; // edi
  unsigned int v24; // esi
  char *v25; // rax
  char *v26; // r11
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  char *v30; // r9
  char v31; // cl
  __int64 v32; // rdx
  int v33; // esi
  char **v34; // rax
  char *v35; // rdi
  __int64 v36; // rdi
  unsigned int v37; // esi
  int v38; // eax
  unsigned int v39; // eax
  char **v40; // r11
  int v41; // edx
  unsigned int v42; // edi
  __int64 v43; // rcx
  int v44; // edi
  __int64 v45; // rax
  char *v46; // rsi
  __int64 v47; // rcx
  int v48; // edi
  __int64 v49; // rax
  char *v50; // rsi
  char **v51; // rdx
  int v52; // eax
  int v53; // eax
  __int64 v54; // [rsp+0h] [rbp-A0h]
  __int64 v55; // [rsp+0h] [rbp-A0h]
  __int64 v56; // [rsp+8h] [rbp-98h]
  _BYTE *v57; // [rsp+8h] [rbp-98h]
  int v58; // [rsp+8h] [rbp-98h]
  _BYTE *v59; // [rsp+8h] [rbp-98h]
  _BYTE *v60; // [rsp+8h] [rbp-98h]
  _BYTE *v61; // [rsp+10h] [rbp-90h]
  __int64 v62; // [rsp+10h] [rbp-90h]
  unsigned int v63; // [rsp+10h] [rbp-90h]
  int v64; // [rsp+10h] [rbp-90h]
  char *v65; // [rsp+10h] [rbp-90h]
  char *v66; // [rsp+10h] [rbp-90h]
  int v67; // [rsp+10h] [rbp-90h]
  int v68; // [rsp+10h] [rbp-90h]
  _BYTE *v69; // [rsp+20h] [rbp-80h] BYREF
  __int64 v70; // [rsp+28h] [rbp-78h]
  _BYTE v71[112]; // [rsp+30h] [rbp-70h] BYREF

  v5 = a1;
  v69 = v71;
  v70 = 0x800000000LL;
  v7 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) == 0 )
  {
    v9 = (char **)a1;
    v8 = (char **)(a1 - v7 * 8);
    if ( a1 - v7 * 8 != a1 )
      goto LABEL_3;
LABEL_25:
    v16 = 0;
    v17 = v71;
    if ( *(_BYTE *)a1 == 5 )
      goto LABEL_9;
    goto LABEL_26;
  }
  v8 = *(char ***)(a1 - 8);
  v9 = &v8[v7];
  if ( v8 == &v8[v7] )
    goto LABEL_25;
LABEL_3:
  v10 = (_BYTE *)a1;
  do
  {
    v12 = *v8;
    v13 = **v8;
    if ( v13 == 11 || v13 == 5 )
    {
      v21 = *(_BYTE *)(a4 + 8) & 1;
      if ( v21 )
      {
        v22 = (char *)(a4 + 16);
        v23 = 3;
      }
      else
      {
        v28 = *(unsigned int *)(a4 + 24);
        v22 = *(char **)(a4 + 16);
        if ( !(_DWORD)v28 )
          goto LABEL_35;
        v23 = v28 - 1;
      }
      v24 = v23 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v25 = &v22[16 * v24];
      v26 = *(char **)v25;
      if ( v12 == *(char **)v25 )
        goto LABEL_17;
      v38 = 1;
      while ( v26 != (char *)-4096LL )
      {
        v24 = v23 & (v38 + v24);
        v64 = v38 + 1;
        v25 = &v22[16 * v24];
        v26 = *(char **)v25;
        if ( v12 == *(char **)v25 )
          goto LABEL_17;
        v38 = v64;
      }
      if ( v21 )
      {
        v36 = 64;
        goto LABEL_36;
      }
      v28 = *(unsigned int *)(a4 + 24);
LABEL_35:
      v36 = 16 * v28;
LABEL_36:
      v25 = &v22[v36];
LABEL_17:
      v27 = 64;
      if ( !v21 )
        v27 = 16LL * *(unsigned int *)(a4 + 24);
      if ( v25 != &v22[v27] )
      {
        v12 = (char *)*((_QWORD *)v25 + 1);
        v14 = (unsigned int)v70;
        v15 = (unsigned int)v70 + 1LL;
        if ( v15 <= HIDWORD(v70) )
          goto LABEL_7;
        goto LABEL_21;
      }
      if ( v13 != 11 )
        v22 = *v8;
      v57 = v10;
      v62 = a3;
      v29 = sub_97B040(v12, a2, a3, a4, v10, v22);
      v10 = v57;
      a3 = v62;
      v30 = (char *)v29;
      v31 = *(_BYTE *)(a4 + 8) & 1;
      if ( v31 )
      {
        v32 = a4 + 16;
        v33 = 3;
      }
      else
      {
        v37 = *(_DWORD *)(a4 + 24);
        v32 = *(_QWORD *)(a4 + 16);
        if ( !v37 )
        {
          v39 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v40 = 0;
          v41 = (v39 >> 1) + 1;
          goto LABEL_44;
        }
        v33 = v37 - 1;
      }
      v63 = v33 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v34 = (char **)(v32 + 16LL * v63);
      v35 = *v34;
      if ( v12 == *v34 )
      {
LABEL_33:
        v12 = v30;
        goto LABEL_6;
      }
      v58 = 1;
      v40 = 0;
      while ( v35 != (char *)-4096LL )
      {
        if ( v40 || v35 != (char *)-8192LL )
          v34 = v40;
        v63 = v33 & (v63 + v58);
        v35 = *(char **)(v32 + 16LL * v63);
        if ( v12 == v35 )
          goto LABEL_33;
        ++v58;
        v40 = v34;
        v34 = (char **)(v32 + 16LL * v63);
      }
      if ( !v40 )
        v40 = v34;
      v39 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v41 = (v39 >> 1) + 1;
      if ( v31 )
      {
        v42 = 12;
        v37 = 4;
LABEL_45:
        if ( v42 <= 4 * v41 )
        {
          v54 = a3;
          v59 = v10;
          v65 = v30;
          sub_97AC20(a4, 2 * v37);
          v30 = v65;
          v10 = v59;
          a3 = v54;
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v43 = a4 + 16;
            v44 = 3;
          }
          else
          {
            v52 = *(_DWORD *)(a4 + 24);
            v43 = *(_QWORD *)(a4 + 16);
            if ( !v52 )
              goto LABEL_91;
            v44 = v52 - 1;
          }
          v45 = v44 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v40 = (char **)(v43 + 16 * v45);
          v46 = *v40;
          if ( v12 == *v40 )
            goto LABEL_61;
          v68 = 1;
          v51 = 0;
          while ( v46 != (char *)-4096LL )
          {
            if ( !v51 && v46 == (char *)-8192LL )
              v51 = v40;
            LODWORD(v45) = v44 & (v68 + v45);
            v40 = (char **)(v43 + 16LL * (unsigned int)v45);
            v46 = *v40;
            if ( v12 == *v40 )
              goto LABEL_61;
            ++v68;
          }
        }
        else
        {
          if ( v37 - *(_DWORD *)(a4 + 12) - v41 > v37 >> 3 )
          {
LABEL_47:
            *(_DWORD *)(a4 + 8) = (2 * (v39 >> 1) + 2) | v39 & 1;
            if ( *v40 != (char *)-4096LL )
              --*(_DWORD *)(a4 + 12);
            *v40 = v12;
            v40[1] = v30;
            goto LABEL_33;
          }
          v55 = a3;
          v60 = v10;
          v66 = v30;
          sub_97AC20(a4, v37);
          v30 = v66;
          v10 = v60;
          a3 = v55;
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v47 = a4 + 16;
            v48 = 3;
          }
          else
          {
            v53 = *(_DWORD *)(a4 + 24);
            v47 = *(_QWORD *)(a4 + 16);
            if ( !v53 )
            {
LABEL_91:
              *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
              BUG();
            }
            v48 = v53 - 1;
          }
          v49 = v48 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v40 = (char **)(v47 + 16 * v49);
          v50 = *v40;
          if ( v12 == *v40 )
          {
LABEL_61:
            v39 = *(_DWORD *)(a4 + 8);
            goto LABEL_47;
          }
          v67 = 1;
          v51 = 0;
          while ( v50 != (char *)-4096LL )
          {
            if ( !v51 && v50 == (char *)-8192LL )
              v51 = v40;
            LODWORD(v49) = v48 & (v67 + v49);
            v40 = (char **)(v47 + 16LL * (unsigned int)v49);
            v50 = *v40;
            if ( v12 == *v40 )
              goto LABEL_61;
            ++v67;
          }
        }
        if ( v51 )
          v40 = v51;
        goto LABEL_61;
      }
      v37 = *(_DWORD *)(a4 + 24);
LABEL_44:
      v42 = 3 * v37;
      goto LABEL_45;
    }
LABEL_6:
    v14 = (unsigned int)v70;
    v15 = (unsigned int)v70 + 1LL;
    if ( v15 <= HIDWORD(v70) )
      goto LABEL_7;
LABEL_21:
    v56 = a3;
    v61 = v10;
    sub_C8D5F0(&v69, v71, v15, 8);
    v14 = (unsigned int)v70;
    a3 = v56;
    v10 = v61;
LABEL_7:
    v8 += 4;
    *(_QWORD *)&v69[8 * v14] = v12;
    v16 = (unsigned int)(v70 + 1);
    LODWORD(v70) = v70 + 1;
  }
  while ( v9 != v8 );
  v5 = (__int64)v10;
  v17 = v69;
  if ( *v10 != 5 )
  {
LABEL_26:
    v18 = v16;
    v19 = sub_AD3730(v17, v16);
    goto LABEL_11;
  }
LABEL_9:
  v18 = *(unsigned __int16 *)(v5 + 2);
  v19 = sub_97CCD0(v5, 1);
  if ( !v19 )
    v19 = v5;
LABEL_11:
  if ( v69 != v71 )
    _libc_free(v69, v18);
  return v19;
}
