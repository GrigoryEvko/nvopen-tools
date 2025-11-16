// Function: sub_35A5FD0
// Address: 0x35a5fd0
//
void __fastcall sub_35A5FD0(int *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 v12; // rbx
  int v13; // r12d
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 i; // rbx
  unsigned __int64 v18; // rsi
  _BYTE *v19; // rax
  _BYTE *v20; // rdx
  __int64 v21; // r12
  int v22; // edx
  int v23; // eax
  __int64 *v24; // rbx
  __int64 v25; // rsi
  int v26; // edx
  bool v27; // cl
  __int64 v28; // r8
  __int64 v29; // rax
  unsigned int v30; // edi
  __int64 *v31; // rdx
  __int64 v32; // r9
  int v33; // r13d
  _QWORD *v34; // rax
  bool v35; // cl
  __int64 v36; // r12
  int v37; // eax
  int v38; // r11d
  __int64 *v39; // rcx
  unsigned int v40; // r8d
  __int64 *v41; // rax
  __int64 v42; // rdi
  _DWORD *v43; // rax
  __int64 v44; // r13
  __int64 v45; // rdx
  _BYTE *v46; // rbx
  _BYTE *v47; // r12
  __int64 v48; // rsi
  __int64 v49; // rdi
  int v50; // eax
  int v51; // edx
  __int64 *v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rdi
  int v55; // r11d
  __int64 *v56; // r10
  int v57; // r11d
  unsigned int v58; // edx
  __int64 v59; // rdi
  __int64 *v60; // rdx
  __int64 *v61; // r12
  __int64 v62; // rsi
  __int64 *v63; // rbx
  int v64; // ecx
  int v65; // edx
  int v66; // r11d
  bool v69; // [rsp+34h] [rbp-ACh]
  __int64 v70; // [rsp+40h] [rbp-A0h]
  __int64 *v71; // [rsp+48h] [rbp-98h]
  __int64 v72; // [rsp+58h] [rbp-88h] BYREF
  __int64 v73; // [rsp+60h] [rbp-80h] BYREF
  __int64 *v74; // [rsp+68h] [rbp-78h]
  __int64 v75; // [rsp+70h] [rbp-70h]
  unsigned int v76; // [rsp+78h] [rbp-68h]
  _BYTE *v77; // [rsp+80h] [rbp-60h] BYREF
  __int64 v78; // [rsp+88h] [rbp-58h]
  _BYTE v79[32]; // [rsp+90h] [rbp-50h] BYREF
  _BYTE v80[48]; // [rsp+B0h] [rbp-30h] BYREF

  v8 = *(_QWORD *)a3;
  v9 = *(_QWORD *)a3 + 32LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v9 )
  {
    do
    {
      v10 = *(unsigned int *)(v9 - 8);
      v11 = *(_QWORD *)(v9 - 24);
      v9 -= 32;
      sub_C7D6A0(v11, 8 * v10, 4);
    }
    while ( v8 != v9 );
  }
  *(_DWORD *)(a3 + 8) = 0;
  v12 = a1[32];
  v13 = a1[32];
  if ( !v13 )
  {
    v77 = v79;
    v78 = 0x100000000LL;
LABEL_88:
    v73 = 0;
    v23 = 0;
    v74 = 0;
    v75 = 0;
    v76 = 0;
    goto LABEL_34;
  }
  v14 = *(unsigned int *)(a3 + 12);
  v15 = 0;
  if ( v12 > v14 )
  {
    sub_359C370(a3, a1[32], v14, a4, a5, a6);
    v15 = 32LL * *(unsigned int *)(a3 + 8);
  }
  v16 = *(_QWORD *)a3 + v15;
  for ( i = *(_QWORD *)a3 + 32 * v12; i != v16; v16 += 32 )
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = 0;
      *(_DWORD *)(v16 + 24) = 0;
      *(_QWORD *)(v16 + 8) = 0;
      *(_DWORD *)(v16 + 16) = 0;
      *(_DWORD *)(v16 + 20) = 0;
    }
  }
  *(_DWORD *)(a3 + 8) = v13;
  v18 = a1[32];
  v77 = v79;
  v78 = 0x100000000LL;
  if ( !v18 )
    goto LABEL_88;
  v19 = v79;
  v20 = v80;
  v21 = 32 * v18;
  if ( v18 == 1
    || (sub_359C370((__int64)&v77, v18, (__int64)v80, a4, a5, a6),
        v20 = &v77[v21],
        v19 = &v77[32 * (unsigned int)v78],
        v19 != &v77[v21]) )
  {
    do
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = 0;
        *((_DWORD *)v19 + 6) = 0;
        *((_QWORD *)v19 + 1) = 0;
        *((_DWORD *)v19 + 4) = 0;
        *((_DWORD *)v19 + 5) = 0;
      }
      v19 += 32;
    }
    while ( v20 != v19 );
  }
  v22 = a1[32];
  LODWORD(v78) = v18;
  v73 = 0;
  v74 = 0;
  v23 = v22;
  v75 = 0;
  v76 = 0;
  if ( v22 <= 0 )
    goto LABEL_34;
  v70 = 0;
  do
  {
    v24 = *(__int64 **)(*(_QWORD *)a1 + 8LL);
    v71 = *(__int64 **)(*(_QWORD *)a1 + 16LL);
    if ( v24 != v71 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v25 = *v24;
          v26 = *(unsigned __int16 *)(*v24 + 68);
          v72 = *v24;
          v27 = v26 == 0 || v26 == 68;
          if ( !v27 )
            break;
          if ( v71 == ++v24 )
            goto LABEL_31;
        }
        v28 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
        v29 = *(unsigned int *)(*(_QWORD *)a1 + 88LL);
        if ( (_DWORD)v29 )
        {
          v30 = (v29 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v31 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v31;
          if ( v25 == *v31 )
          {
LABEL_23:
            if ( v31 != (__int64 *)(v28 + 16 * v29) )
            {
              v33 = *((_DWORD *)v31 + 2);
              v27 = v33 == 0;
              goto LABEL_25;
            }
          }
          else
          {
            v51 = 1;
            while ( v32 != -4096 )
            {
              v66 = v51 + 1;
              v30 = (v29 - 1) & (v51 + v30);
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( v25 == *v31 )
                goto LABEL_23;
              v51 = v66;
            }
          }
        }
        v33 = -1;
LABEL_25:
        v69 = v27;
        v34 = sub_359A2A0((__int64)a1, v25);
        v35 = v69;
        v36 = (__int64)v34;
        v37 = a1[32] - 1;
        if ( v37 == (_DWORD)v70 )
        {
          v52 = sub_359C4A0(a4, &v72);
          v35 = v69;
          *v52 = v36;
          v37 = a1[32] - 1;
        }
        sub_359F7A0((__int64 *)a1, v36, *(_QWORD *)a3 + 32 * v70, (_DWORD)v70 == v37 && v35);
        sub_359CE40((__int64)a1, v72, v70, a2, (_QWORD *)a3, &v77);
        if ( !v76 )
        {
          ++v73;
          goto LABEL_57;
        }
        v38 = 1;
        v39 = 0;
        v40 = (v76 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v41 = &v74[2 * v40];
        v42 = *v41;
        if ( v36 != *v41 )
        {
          while ( v42 != -4096 )
          {
            if ( !v39 && v42 == -8192 )
              v39 = v41;
            v40 = (v76 - 1) & (v38 + v40);
            v41 = &v74[2 * v40];
            v42 = *v41;
            if ( v36 == *v41 )
              goto LABEL_29;
            ++v38;
          }
          if ( !v39 )
            v39 = v41;
          ++v73;
          v50 = v75 + 1;
          if ( 4 * ((int)v75 + 1) >= 3 * v76 )
          {
LABEL_57:
            sub_35A57D0((__int64)&v73, 2 * v76);
            if ( !v76 )
              goto LABEL_96;
            v53 = (v76 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v50 = v75 + 1;
            v39 = &v74[2 * v53];
            v54 = *v39;
            if ( v36 != *v39 )
            {
              v55 = 1;
              v56 = 0;
              while ( v54 != -4096 )
              {
                if ( !v56 && v54 == -8192 )
                  v56 = v39;
                v53 = (v76 - 1) & (v55 + v53);
                v39 = &v74[2 * v53];
                v54 = *v39;
                if ( v36 == *v39 )
                  goto LABEL_50;
                ++v55;
              }
LABEL_61:
              if ( v56 )
                v39 = v56;
            }
          }
          else if ( v76 - HIDWORD(v75) - v50 <= v76 >> 3 )
          {
            sub_35A57D0((__int64)&v73, v76);
            if ( !v76 )
            {
LABEL_96:
              LODWORD(v75) = v75 + 1;
              BUG();
            }
            v56 = 0;
            v57 = 1;
            v58 = (v76 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v50 = v75 + 1;
            v39 = &v74[2 * v58];
            v59 = *v39;
            if ( v36 != *v39 )
            {
              while ( v59 != -4096 )
              {
                if ( !v56 && v59 == -8192 )
                  v56 = v39;
                v58 = (v76 - 1) & (v57 + v58);
                v39 = &v74[2 * v58];
                v59 = *v39;
                if ( v36 == *v39 )
                  goto LABEL_50;
                ++v57;
              }
              goto LABEL_61;
            }
          }
LABEL_50:
          LODWORD(v75) = v50;
          if ( *v39 != -4096 )
            --HIDWORD(v75);
          *v39 = v36;
          v43 = v39 + 1;
          v39[1] = 0;
          goto LABEL_30;
        }
LABEL_29:
        v43 = v41 + 1;
LABEL_30:
        *v43 = v70;
        ++v24;
        v43[1] = v33;
        v44 = *((_QWORD *)a1 + 11);
        sub_2E31040((__int64 *)(v44 + 40), v36);
        v45 = *(_QWORD *)(v44 + 48);
        *(_QWORD *)(v36 + 8) = v44 + 48;
        v45 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v36 = v45 | *(_QWORD *)v36 & 7LL;
        *(_QWORD *)(v45 + 8) = v36;
        *(_QWORD *)(v44 + 48) = *(_QWORD *)(v44 + 48) & 7LL | v36;
        if ( v71 == v24 )
        {
LABEL_31:
          v22 = a1[32];
          break;
        }
      }
    }
    ++v70;
    v23 = v22;
  }
  while ( v22 > (int)v70 );
  if ( (_DWORD)v75 )
  {
    v60 = v74;
    v61 = &v74[2 * v76];
    if ( v74 != v61 )
    {
      while ( 1 )
      {
        v62 = *v60;
        v63 = v60;
        if ( *v60 != -8192 && v62 != -4096 )
          break;
        v60 += 2;
        if ( v61 == v60 )
          goto LABEL_34;
      }
      if ( v60 != v61 )
      {
        do
        {
          v64 = *((_DWORD *)v63 + 2);
          v65 = *((_DWORD *)v63 + 3);
          v63 += 2;
          sub_359CAC0((__int64 *)a1, v62, v65, v64, (_QWORD *)a3, (__int64)&v77);
          if ( v63 == v61 )
            break;
          while ( 1 )
          {
            v62 = *v63;
            if ( *v63 != -8192 && v62 != -4096 )
              break;
            v63 += 2;
            if ( v61 == v63 )
              goto LABEL_84;
          }
        }
        while ( v63 != v61 );
LABEL_84:
        v23 = a1[32];
      }
    }
  }
LABEL_34:
  sub_359A2D0((__int64)a1, *((_QWORD *)a1 + 11), v23 - 1, a4, *((_QWORD *)a1 + 11), *((_QWORD *)a1 + 12));
  sub_C7D6A0(0, 0, 8);
  sub_C7D6A0((__int64)v74, 16LL * v76, 8);
  v46 = v77;
  v47 = &v77[32 * (unsigned int)v78];
  if ( v77 != v47 )
  {
    do
    {
      v48 = *((unsigned int *)v47 - 2);
      v49 = *((_QWORD *)v47 - 3);
      v47 -= 32;
      sub_C7D6A0(v49, 8 * v48, 4);
    }
    while ( v46 != v47 );
    v47 = v77;
  }
  if ( v47 != v79 )
    _libc_free((unsigned __int64)v47);
}
