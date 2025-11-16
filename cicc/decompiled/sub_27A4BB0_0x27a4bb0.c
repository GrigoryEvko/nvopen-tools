// Function: sub_27A4BB0
// Address: 0x27a4bb0
//
__int64 __fastcall sub_27A4BB0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 *v4; // r10
  __int64 v5; // r12
  __int64 *v6; // rcx
  char **v7; // rax
  char *v8; // rbx
  char *v9; // rdx
  int v10; // edi
  __int64 v11; // r9
  int v12; // edi
  unsigned int v13; // r8d
  __int64 v14; // rsi
  char *v15; // r13
  unsigned int v16; // r13d
  unsigned int v17; // r8d
  __int64 v18; // rsi
  char *v19; // r11
  unsigned int v20; // r13d
  char v21; // al
  __int64 v22; // rbx
  unsigned __int64 v24; // rax
  int v25; // edx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // esi
  int v30; // r10d
  __int64 v31; // r8
  unsigned __int64 *v32; // rax
  __int64 v33; // rcx
  __int64 *v34; // rdx
  __int64 v35; // rdi
  int v36; // r13d
  unsigned int v37; // esi
  __int64 v38; // r8
  int v39; // r10d
  char **v40; // rax
  __int64 v41; // rcx
  _QWORD *v42; // rdx
  char *v43; // rdi
  _DWORD *v44; // rax
  int v45; // esi
  int v46; // esi
  int v47; // r11d
  int v48; // edi
  int v49; // ecx
  int v50; // edi
  int v51; // ecx
  _DWORD *v52; // rax
  int v53; // r11d
  int v54; // r11d
  __int64 v55; // r10
  unsigned int v56; // r8d
  char *v57; // rsi
  int v58; // r9d
  char **v59; // rdx
  int v60; // r11d
  int v61; // r11d
  __int64 v62; // r10
  unsigned int v63; // r8d
  unsigned __int64 v64; // rsi
  int v65; // r9d
  unsigned __int64 *v66; // rdx
  int v67; // r10d
  int v68; // r10d
  int v69; // r8d
  __int64 v70; // r9
  unsigned int v71; // r11d
  unsigned __int64 v72; // rsi
  int v73; // r10d
  int v74; // r10d
  int v75; // r8d
  __int64 v76; // r9
  unsigned int v77; // r11d
  char *v78; // rsi
  int v79; // [rsp+Ch] [rbp-64h]
  unsigned int v80; // [rsp+Ch] [rbp-64h]
  unsigned int v81; // [rsp+Ch] [rbp-64h]
  __int64 v82; // [rsp+10h] [rbp-60h]
  __int64 v83; // [rsp+18h] [rbp-58h]
  int v84; // [rsp+20h] [rbp-50h]
  int v85; // [rsp+24h] [rbp-4Ch]
  unsigned int v86; // [rsp+28h] [rbp-48h]
  int v87; // [rsp+2Ch] [rbp-44h]
  __int64 v88; // [rsp+30h] [rbp-40h]
  __int64 v89; // [rsp+38h] [rbp-38h]

  v88 = *a2 + 56LL * *((unsigned int *)a2 + 2);
  if ( *a2 == v88 )
  {
    v86 = 0;
    v22 = 0;
    goto LABEL_22;
  }
  v86 = 0;
  v3 = *a2;
  v84 = 0;
  v85 = 0;
  v87 = 0;
  v82 = a1 + 264;
  do
  {
    v4 = *(__int64 **)(v3 + 8);
    v5 = *(_QWORD *)v3;
    v89 = v3 + 8;
    v6 = &v4[*(unsigned int *)(v3 + 16)];
    if ( v4 == v6 )
      goto LABEL_27;
    v7 = *(char ***)(v3 + 8);
    v8 = 0;
    do
    {
      while ( 1 )
      {
        v9 = *v7;
        if ( v5 == *((_QWORD *)*v7 + 5) )
        {
          if ( !v8 )
          {
            v8 = *v7;
            goto LABEL_5;
          }
          v10 = *(_DWORD *)(a1 + 288);
          v11 = *(_QWORD *)(a1 + 272);
          if ( v10 )
            break;
        }
LABEL_5:
        if ( v6 == (__int64 *)++v7 )
          goto LABEL_15;
      }
      v12 = v10 - 1;
      v13 = v12 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v14 = v11 + 16LL * v13;
      v15 = *(char **)v14;
      if ( v9 == *(char **)v14 )
      {
LABEL_10:
        v16 = *(_DWORD *)(v14 + 8);
      }
      else
      {
        v46 = 1;
        while ( v15 != (char *)-4096LL )
        {
          v47 = v46 + 1;
          v13 = v12 & (v46 + v13);
          v14 = v11 + 16LL * v13;
          v15 = *(char **)v14;
          if ( v9 == *(char **)v14 )
            goto LABEL_10;
          v46 = v47;
        }
        v16 = 0;
      }
      v17 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v18 = v11 + 16LL * v17;
      v19 = *(char **)v18;
      if ( *(char **)v18 != v8 )
      {
        v45 = 1;
        while ( v19 != (char *)-4096LL )
        {
          v17 = v12 & (v45 + v17);
          v79 = v45 + 1;
          v18 = v11 + 16LL * v17;
          v19 = *(char **)v18;
          if ( *(char **)v18 == v8 )
            goto LABEL_12;
          v45 = v79;
        }
        goto LABEL_5;
      }
LABEL_12:
      if ( *(_DWORD *)(v18 + 8) > v16 )
        v8 = *v7;
      ++v7;
    }
    while ( v6 != (__int64 *)v7 );
LABEL_15:
    v20 = 0;
    if ( v8 )
    {
LABEL_16:
      sub_AE8F80(v8);
      sub_27A3590(a1, v89, (unsigned __int8 *)v8, v5, v20);
      v21 = *v8;
      if ( *v8 == 61 )
      {
        ++v87;
      }
      else if ( v21 == 62 )
      {
        ++v85;
      }
      else if ( v21 == 85 )
      {
        ++v84;
      }
      else
      {
        ++v86;
      }
      goto LABEL_20;
    }
LABEL_27:
    v8 = (char *)*v4;
    if ( (unsigned __int8)sub_27A2D20(a1, *v4, v5)
      || !*(_BYTE *)(a1 + 636) && (unsigned __int8)sub_27A3680(a1, v8, v5, v89) )
    {
      v24 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v24 == v5 + 48 )
      {
        v26 = 0;
      }
      else
      {
        if ( !v24 )
          BUG();
        v25 = *(unsigned __int8 *)(v24 - 24);
        v26 = 0;
        v27 = v24 - 24;
        if ( (unsigned int)(v25 - 30) < 0xB )
          v26 = v27;
      }
      sub_1031600(*(_QWORD *)(a1 + 240), (__int64)v8);
      v28 = v83;
      LOWORD(v28) = 0;
      v83 = v28;
      sub_B444E0(v8, v26 + 24, v28);
      v29 = *(_DWORD *)(a1 + 288);
      if ( v29 )
      {
        v30 = 1;
        v31 = *(_QWORD *)(a1 + 272);
        v32 = 0;
        v33 = (v29 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v34 = (__int64 *)(v31 + 16 * v33);
        v35 = *v34;
        if ( v26 == *v34 )
        {
LABEL_36:
          v36 = *((_DWORD *)v34 + 2);
          *((_DWORD *)v34 + 2) = v36 + 1;
          v37 = *(_DWORD *)(a1 + 288);
          if ( v37 )
            goto LABEL_37;
LABEL_80:
          ++*(_QWORD *)(a1 + 264);
LABEL_81:
          sub_A429D0(v82, 2 * v37);
          v53 = *(_DWORD *)(a1 + 288);
          if ( v53 )
          {
            v54 = v53 - 1;
            v55 = *(_QWORD *)(a1 + 272);
            v56 = v54 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v49 = *(_DWORD *)(a1 + 280) + 1;
            v40 = (char **)(v55 + 16LL * v56);
            v57 = *v40;
            if ( v8 != *v40 )
            {
              v58 = 1;
              v59 = 0;
              while ( v57 != (char *)-4096LL )
              {
                if ( !v59 && v57 == (char *)-8192LL )
                  v59 = v40;
                v56 = v54 & (v58 + v56);
                v40 = (char **)(v55 + 16LL * v56);
                v57 = *v40;
                if ( v8 == *v40 )
                  goto LABEL_64;
                ++v58;
              }
LABEL_109:
              if ( v59 )
                v40 = v59;
            }
            goto LABEL_64;
          }
LABEL_123:
          ++*(_DWORD *)(a1 + 280);
          BUG();
        }
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v32 )
            v32 = (unsigned __int64 *)v34;
          LODWORD(v33) = (v29 - 1) & (v30 + v33);
          v34 = (__int64 *)(v31 + 16LL * (unsigned int)v33);
          v35 = *v34;
          if ( v26 == *v34 )
            goto LABEL_36;
          ++v30;
        }
        v50 = *(_DWORD *)(a1 + 280);
        if ( !v32 )
          v32 = (unsigned __int64 *)v34;
        ++*(_QWORD *)(a1 + 264);
        v51 = v50 + 1;
        if ( 4 * (v50 + 1) < 3 * v29 )
        {
          if ( v29 - *(_DWORD *)(a1 + 284) - v51 > v29 >> 3 )
            goto LABEL_77;
          v80 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
          sub_A429D0(v82, v29);
          v67 = *(_DWORD *)(a1 + 288);
          if ( v67 )
          {
            v68 = v67 - 1;
            v69 = 1;
            v66 = 0;
            v70 = *(_QWORD *)(a1 + 272);
            v71 = v68 & v80;
            v51 = *(_DWORD *)(a1 + 280) + 1;
            v32 = (unsigned __int64 *)(v70 + 16LL * (v68 & v80));
            v72 = *v32;
            if ( *v32 != v26 )
            {
              while ( v72 != -4096 )
              {
                if ( !v66 && v72 == -8192 )
                  v66 = v32;
                v71 = v68 & (v69 + v71);
                v32 = (unsigned __int64 *)(v70 + 16LL * v71);
                v72 = *v32;
                if ( v26 == *v32 )
                  goto LABEL_77;
                ++v69;
              }
              goto LABEL_103;
            }
            goto LABEL_77;
          }
          goto LABEL_122;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 264);
      }
      sub_A429D0(v82, 2 * v29);
      v60 = *(_DWORD *)(a1 + 288);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(a1 + 272);
        v63 = v61 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v51 = *(_DWORD *)(a1 + 280) + 1;
        v32 = (unsigned __int64 *)(v62 + 16LL * v63);
        v64 = *v32;
        if ( v26 != *v32 )
        {
          v65 = 1;
          v66 = 0;
          while ( v64 != -4096 )
          {
            if ( v64 == -8192 && !v66 )
              v66 = v32;
            v63 = v61 & (v65 + v63);
            v32 = (unsigned __int64 *)(v62 + 16LL * v63);
            v64 = *v32;
            if ( v26 == *v32 )
              goto LABEL_77;
            ++v65;
          }
LABEL_103:
          if ( v66 )
            v32 = v66;
        }
LABEL_77:
        *(_DWORD *)(a1 + 280) = v51;
        if ( *v32 != -4096 )
          --*(_DWORD *)(a1 + 284);
        *v32 = v26;
        v52 = v32 + 1;
        v36 = 0;
        *v52 = 0;
        *v52 = 1;
        v37 = *(_DWORD *)(a1 + 288);
        if ( !v37 )
          goto LABEL_80;
LABEL_37:
        v38 = *(_QWORD *)(a1 + 272);
        v39 = 1;
        v40 = 0;
        v41 = (v37 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v42 = (_QWORD *)(v38 + 16 * v41);
        v43 = (char *)*v42;
        if ( v8 == (char *)*v42 )
        {
LABEL_38:
          v44 = v42 + 1;
        }
        else
        {
          while ( v43 != (char *)-4096LL )
          {
            if ( v43 == (char *)-8192LL && !v40 )
              v40 = (char **)v42;
            LODWORD(v41) = (v37 - 1) & (v39 + v41);
            v42 = (_QWORD *)(v38 + 16LL * (unsigned int)v41);
            v43 = (char *)*v42;
            if ( v8 == (char *)*v42 )
              goto LABEL_38;
            ++v39;
          }
          v48 = *(_DWORD *)(a1 + 280);
          if ( !v40 )
            v40 = (char **)v42;
          ++*(_QWORD *)(a1 + 264);
          v49 = v48 + 1;
          if ( 4 * (v48 + 1) >= 3 * v37 )
            goto LABEL_81;
          if ( v37 - *(_DWORD *)(a1 + 284) - v49 > v37 >> 3 )
            goto LABEL_64;
          v81 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
          sub_A429D0(v82, v37);
          v73 = *(_DWORD *)(a1 + 288);
          if ( !v73 )
            goto LABEL_123;
          v74 = v73 - 1;
          v75 = 1;
          v59 = 0;
          v76 = *(_QWORD *)(a1 + 272);
          v77 = v74 & v81;
          v49 = *(_DWORD *)(a1 + 280) + 1;
          v40 = (char **)(v76 + 16LL * (v74 & v81));
          v78 = *v40;
          if ( v8 != *v40 )
          {
            while ( v78 != (char *)-4096LL )
            {
              if ( !v59 && v78 == (char *)-8192LL )
                v59 = v40;
              v77 = v74 & (v75 + v77);
              v40 = (char **)(v76 + 16LL * v77);
              v78 = *v40;
              if ( v8 == *v40 )
                goto LABEL_64;
              ++v75;
            }
            goto LABEL_109;
          }
LABEL_64:
          *(_DWORD *)(a1 + 280) = v49;
          if ( *v40 != (char *)-4096LL )
            --*(_DWORD *)(a1 + 284);
          *v40 = v8;
          v44 = v40 + 1;
          *v44 = 0;
        }
        *v44 = v36;
        v20 = 1;
        goto LABEL_16;
      }
LABEL_122:
      ++*(_DWORD *)(a1 + 280);
      BUG();
    }
LABEL_20:
    v3 += 56;
  }
  while ( v88 != v3 );
  v22 = (unsigned int)(v87 + v84 + v85);
LABEL_22:
  if ( *(_QWORD *)(a1 + 248) && byte_4F8F8E8[0] )
    nullsub_390();
  return (v22 << 32) | v86;
}
