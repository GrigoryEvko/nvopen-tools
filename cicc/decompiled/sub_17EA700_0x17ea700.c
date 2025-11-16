// Function: sub_17EA700
// Address: 0x17ea700
//
void __fastcall sub_17EA700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 *v8; // rbx
  __int64 v9; // r8
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r13
  __int64 v14; // r13
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r11
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  int v22; // eax
  _QWORD *v23; // r15
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // rdi
  _QWORD *v29; // r15
  _QWORD *v30; // r12
  char v31; // di
  _QWORD *v32; // r8
  unsigned int v33; // edx
  _QWORD *v34; // rax
  _QWORD *v35; // r10
  __int64 v36; // r13
  int v37; // r8d
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  __int64 k; // rsi
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned int v43; // r8d
  __int64 *v44; // rdx
  __int64 v45; // r10
  unsigned __int64 v46; // r12
  __int64 v47; // r15
  __int64 v48; // rdx
  __int64 v49; // r13
  __int64 v50; // r11
  __int64 v51; // r9
  int v52; // r14d
  __int64 v53; // rcx
  unsigned int v54; // esi
  __int64 *v55; // rax
  __int64 v56; // r8
  __int64 v57; // rax
  unsigned __int64 v58; // rax
  __int64 v59; // r14
  _QWORD *v60; // r15
  __int64 v61; // r13
  __int64 v62; // rax
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // rsi
  int v66; // eax
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 i; // rsi
  bool v71; // zf
  __int64 v72; // rax
  unsigned __int64 v73; // rcx
  __int64 m; // rsi
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 j; // rsi
  int v79; // eax
  int v80; // r11d
  int v81; // eax
  int v82; // edx
  int v83; // eax
  int v84; // eax
  int v85; // r11d
  int v86; // r10d
  int v87; // r11d
  int v88; // [rsp+4h] [rbp-4Ch]
  __int64 v89; // [rsp+8h] [rbp-48h]
  unsigned __int64 v90; // [rsp+8h] [rbp-48h]
  __int64 v91; // [rsp+8h] [rbp-48h]
  __int64 v92; // [rsp+8h] [rbp-48h]
  unsigned int v93; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v94[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = a1;
  v7 = *(__int64 **)(a1 + 248);
  if ( v7 != *(__int64 **)(a1 + 256) )
  {
    v8 = *(__int64 **)(a1 + 256);
    v9 = a1;
    while ( 1 )
    {
      v23 = (_QWORD *)*v7;
      if ( !*(_BYTE *)(*v7 + 25) )
        break;
LABEL_12:
      if ( v8 == ++v7 )
      {
        v6 = v9;
        goto LABEL_18;
      }
    }
    v24 = *(unsigned int *)(v9 + 296);
    v25 = *(_QWORD *)(v9 + 280);
    if ( (_DWORD)v24 )
    {
      a6 = v24 - 1;
      v10 = v23[1];
      v11 = (v24 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
      v12 = (__int64 *)(v25 + 16LL * v11);
      v13 = *v12;
      if ( *v23 == *v12 )
      {
LABEL_4:
        v14 = v12[1];
      }
      else
      {
        v84 = 1;
        while ( v13 != -8 )
        {
          v87 = v84 + 1;
          v11 = a6 & (v84 + v11);
          v12 = (__int64 *)(v25 + 16LL * v11);
          v13 = *v12;
          if ( *v23 == *v12 )
            goto LABEL_4;
          v84 = v87;
        }
        v14 = *(_QWORD *)(v25 + 16LL * (unsigned int)v24 + 8);
      }
      v15 = a6 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v16 = (__int64 *)(v25 + 16LL * v15);
      v17 = *v16;
      if ( v10 == *v16 )
      {
LABEL_6:
        v18 = v16[1];
      }
      else
      {
        v83 = 1;
        while ( v17 != -8 )
        {
          v86 = v83 + 1;
          v15 = a6 & (v83 + v15);
          v16 = (__int64 *)(v25 + 16LL * v15);
          v17 = *v16;
          if ( v10 == *v16 )
            goto LABEL_6;
          v83 = v86;
        }
        v18 = *(_QWORD *)(v25 + 16 * v24 + 8);
      }
      v19 = *(unsigned int *)(v14 + 80);
      if ( (unsigned int)v19 < *(_DWORD *)(v14 + 84) )
        goto LABEL_8;
    }
    else
    {
      v14 = *(_QWORD *)(v25 + 8);
      v19 = *(unsigned int *)(v14 + 80);
      v18 = v14;
      if ( (unsigned int)v19 < *(_DWORD *)(v14 + 84) )
      {
LABEL_8:
        *(_QWORD *)(*(_QWORD *)(v14 + 72) + 8 * v19) = v23;
        ++*(_DWORD *)(v14 + 80);
        v20 = *v7;
        v21 = *(unsigned int *)(v18 + 48);
        if ( (unsigned int)v21 >= *(_DWORD *)(v18 + 52) )
        {
          v92 = v9;
          sub_16CD150(v18 + 40, (const void *)(v18 + 56), 0, 8, v9, a6);
          v21 = *(unsigned int *)(v18 + 48);
          v9 = v92;
        }
        *(_QWORD *)(*(_QWORD *)(v18 + 40) + 8 * v21) = v20;
        ++*(_DWORD *)(v18 + 48);
        ++*(_DWORD *)(v14 + 32);
        v22 = *(_DWORD *)(v18 + 28);
        *(_DWORD *)(v18 + 28) = v22 + 1;
        if ( *(_BYTE *)(*v7 + 27) )
        {
          *(_DWORD *)(v18 + 28) = v22;
          --*(_DWORD *)(v14 + 32);
        }
        goto LABEL_12;
      }
    }
    v89 = v9;
    sub_16CD150(v14 + 72, (const void *)(v14 + 88), 0, 8, v9, a6);
    v19 = *(unsigned int *)(v14 + 80);
    v9 = v89;
    goto LABEL_8;
  }
LABEL_18:
  v26 = *(_QWORD *)(v6 + 280);
  v27 = *(_DWORD *)(v6 + 296);
  do
  {
    v28 = *(_QWORD *)v6;
    v29 = (_QWORD *)(*(_QWORD *)v6 + 72LL);
    v30 = (_QWORD *)(*v29 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v29 == v30 )
      goto LABEL_41;
    v31 = 0;
    do
    {
      v32 = v30 - 3;
      if ( !v30 )
        v32 = 0;
      if ( v27 )
      {
        v33 = (v27 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
        v34 = (_QWORD *)(v26 + 16LL * v33);
        v35 = (_QWORD *)*v34;
        if ( v32 == (_QWORD *)*v34 )
        {
LABEL_25:
          if ( v34 == (_QWORD *)(v26 + 16LL * v27) )
            goto LABEL_38;
          v36 = v34[1];
          if ( !v36 )
            goto LABEL_38;
          v37 = *(_DWORD *)(v36 + 32);
          if ( *(_BYTE *)(v36 + 24) )
          {
            if ( v37 != 1 )
              goto LABEL_36;
          }
          else
          {
            if ( !v37 )
            {
              v68 = *(_QWORD *)(v36 + 72);
              v69 = 0;
              for ( i = v68 + 8LL * *(unsigned int *)(v36 + 80); i != v68; v68 += 8 )
              {
                if ( !*(_BYTE *)(*(_QWORD *)v68 + 25LL) )
                  v69 += *(_QWORD *)(*(_QWORD *)v68 + 32LL);
              }
              v71 = *(_DWORD *)(v36 + 28) == 1;
              *(_QWORD *)(v36 + 16) = v69;
              v31 = 1;
              *(_BYTE *)(v36 + 24) = 1;
              if ( !v71 )
                goto LABEL_37;
              goto LABEL_75;
            }
            if ( *(_DWORD *)(v36 + 28) )
              goto LABEL_38;
            v76 = *(_QWORD *)(v36 + 40);
            v77 = 0;
            for ( j = v76 + 8LL * *(unsigned int *)(v36 + 48); v76 != j; v76 += 8 )
            {
              if ( !*(_BYTE *)(*(_QWORD *)v76 + 25LL) )
                v77 += *(_QWORD *)(*(_QWORD *)v76 + 32LL);
            }
            *(_QWORD *)(v36 + 16) = v77;
            *(_BYTE *)(v36 + 24) = 1;
            if ( v37 != 1 )
            {
              v31 = 1;
LABEL_37:
              v26 = *(_QWORD *)(v6 + 280);
              v27 = *(_DWORD *)(v6 + 296);
              goto LABEL_38;
            }
          }
          v38 = *(_QWORD *)(v36 + 72);
          v39 = 0;
          for ( k = v38 + 8LL * *(unsigned int *)(v36 + 80); v38 != k; v38 += 8 )
          {
            if ( !*(_BYTE *)(*(_QWORD *)v38 + 25LL) )
              v39 += *(_QWORD *)(*(_QWORD *)v38 + 32LL);
          }
          v41 = *(_QWORD *)(v36 + 16) - v39;
          if ( *(_QWORD *)(v36 + 16) <= v39 )
            v41 = 0;
          sub_17E67C0(v6, (__int64 **)(v36 + 72), v41);
          v31 = 1;
LABEL_36:
          if ( *(_DWORD *)(v36 + 28) != 1 )
            goto LABEL_37;
LABEL_75:
          v72 = *(_QWORD *)(v36 + 40);
          v73 = 0;
          for ( m = v72 + 8LL * *(unsigned int *)(v36 + 48); m != v72; v72 += 8 )
          {
            if ( !*(_BYTE *)(*(_QWORD *)v72 + 25LL) )
              v73 += *(_QWORD *)(*(_QWORD *)v72 + 32LL);
          }
          v75 = *(_QWORD *)(v36 + 16) - v73;
          if ( *(_QWORD *)(v36 + 16) <= v73 )
            v75 = 0;
          sub_17E67C0(v6, (__int64 **)(v36 + 40), v75);
          v27 = *(_DWORD *)(v6 + 296);
          v31 = 1;
          v26 = *(_QWORD *)(v6 + 280);
          goto LABEL_38;
        }
        v79 = 1;
        while ( v35 != (_QWORD *)-8LL )
        {
          v80 = v79 + 1;
          v33 = (v27 - 1) & (v79 + v33);
          v34 = (_QWORD *)(v26 + 16LL * v33);
          v35 = (_QWORD *)*v34;
          if ( v32 == (_QWORD *)*v34 )
            goto LABEL_25;
          v79 = v80;
        }
      }
LABEL_38:
      v30 = (_QWORD *)(*v30 & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( v29 != v30 );
  }
  while ( v31 );
  v28 = *(_QWORD *)v6;
LABEL_41:
  v42 = *(_QWORD *)(v28 + 80);
  if ( v42 )
    v42 -= 24;
  if ( v27 )
  {
    v43 = (v27 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
    v44 = (__int64 *)(v26 + 16LL * v43);
    v45 = *v44;
    if ( v42 == *v44 )
      goto LABEL_45;
    v82 = 1;
    while ( v45 != -8 )
    {
      v85 = v82 + 1;
      v43 = (v27 - 1) & (v82 + v43);
      v44 = (__int64 *)(v26 + 16LL * v43);
      v45 = *v44;
      if ( v42 == *v44 )
        goto LABEL_45;
      v82 = v85;
    }
  }
  v44 = (__int64 *)(v26 + 16LL * v27);
LABEL_45:
  v46 = *(_QWORD *)(v44[1] + 16);
  v90 = v46;
  sub_15E4450(v28, v46, 1, 0);
  v47 = *(_QWORD *)v6;
  v48 = *(_QWORD *)(*(_QWORD *)v6 + 80LL);
  v49 = *(_QWORD *)v6 + 72LL;
  if ( v49 != v48 )
  {
    v50 = *(_QWORD *)(v6 + 280);
    v51 = *(unsigned int *)(v6 + 296);
    v52 = v51 - 1;
    do
    {
      v53 = v48 - 24;
      if ( !v48 )
        v53 = 0;
      if ( (_DWORD)v51 )
      {
        v54 = v52 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
        v55 = (__int64 *)(v50 + 16LL * v54);
        v56 = *v55;
        if ( v53 == *v55 )
        {
LABEL_51:
          if ( (__int64 *)(v50 + 16 * v51) != v55 )
          {
            v57 = v55[1];
            if ( v57 )
            {
              v58 = *(_QWORD *)(v57 + 16);
              if ( v46 < v58 )
                v46 = v58;
            }
          }
        }
        else
        {
          v81 = 1;
          while ( v56 != -8 )
          {
            v54 = v52 & (v81 + v54);
            v88 = v81 + 1;
            v55 = (__int64 *)(v50 + 16LL * v54);
            v56 = *v55;
            if ( v53 == *v55 )
              goto LABEL_51;
            v81 = v88;
          }
        }
      }
      v48 = *(_QWORD *)(v48 + 8);
    }
    while ( v49 != v48 );
  }
  if ( *(_QWORD *)(v6 + 328) )
  {
    sub_16AF710(&v93, 1u, 0x64u);
    sub_16AF710(v94, 2u, 0x2710u);
    if ( v90 < sub_16AF780(&v93, *(_QWORD *)(v6 + 328)) )
    {
      if ( sub_16AF780(v94, *(_QWORD *)(v6 + 328)) >= v46 )
        *(_DWORD *)(v6 + 376) = 1;
    }
    else
    {
      *(_DWORD *)(v6 + 376) = 2;
    }
    v47 = *(_QWORD *)v6;
    v49 = *(_QWORD *)v6 + 72LL;
  }
  *(_DWORD *)(v6 + 76) = 2;
  *(_QWORD *)(v6 + 112) = v6;
  *(_QWORD *)(v6 + 80) = v6 + 336;
  v59 = *(_QWORD *)(v47 + 80);
  if ( v49 != v59 )
  {
    v91 = v49;
    v60 = (_QWORD *)(v6 + 64);
    v61 = v6;
    do
    {
      v62 = v59;
      v59 = *(_QWORD *)(v59 + 8);
      v63 = *(_QWORD *)(v62 + 24);
      v64 = v62 + 16;
LABEL_60:
      while ( v64 != v63 )
      {
        while ( 1 )
        {
          v65 = v63;
          v63 = *(_QWORD *)(v63 + 8);
          if ( *(_BYTE *)(v65 - 8) != 79 || !byte_4FA52E0 || *(_BYTE *)(**(_QWORD **)(v65 - 96) + 8LL) == 16 )
            break;
          v66 = *(_DWORD *)(v61 + 76);
          v67 = v65 - 24;
          if ( v66 == 1 )
          {
            sub_17E2B70((__int64)v60, v67);
            goto LABEL_60;
          }
          if ( v66 == 2 )
          {
            sub_17EA5C0(v60, v67);
            goto LABEL_60;
          }
          ++*(_DWORD *)(v61 + 72);
          if ( v64 == v63 )
            goto LABEL_67;
        }
      }
LABEL_67:
      ;
    }
    while ( v91 != v59 );
  }
}
