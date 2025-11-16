// Function: sub_1A6F940
// Address: 0x1a6f940
//
__int64 __fastcall sub_1A6F940(__int64 a1, char a2)
{
  __int64 **v2; // rbx
  _BYTE *v3; // r15
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rsi
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 *v13; // rax
  __int64 *v14; // r13
  __int64 v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  _BYTE *v23; // rdi
  __int64 v24; // rdx
  char v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rax
  int v30; // r9d
  unsigned int v31; // edx
  __int64 *v32; // rcx
  __int64 v33; // rsi
  __int64 *v34; // rdx
  __int64 *v35; // rax
  unsigned int v36; // esi
  __int64 *v37; // rcx
  __int64 v38; // r10
  __int64 *v39; // rdx
  __int64 *v40; // rcx
  char v41; // cl
  __int64 v42; // rcx
  unsigned __int64 v43; // rax
  __int64 v44; // rcx
  unsigned int v45; // esi
  __int64 v46; // rdi
  __int64 v47; // r8
  __int64 v48; // rdx
  __int64 v49; // rcx
  int v50; // r13d
  __int64 *v51; // r11
  int v52; // edx
  int v53; // ecx
  int v54; // r11d
  int v55; // ecx
  int v56; // r11d
  __int64 *v57; // r9
  int v58; // edx
  int v59; // r8d
  int v60; // r8d
  __int64 v61; // r9
  unsigned int v62; // ecx
  __int64 v63; // r11
  int v64; // edi
  __int64 *v65; // rsi
  int v66; // r8d
  int v67; // r8d
  __int64 v68; // r9
  unsigned int v69; // ecx
  __int64 v70; // r11
  int v71; // edi
  __int64 *v72; // rsi
  int v73; // r8d
  int v74; // r8d
  __int64 v75; // r9
  __int64 *v76; // rdi
  unsigned int v77; // r12d
  int v78; // ecx
  __int64 v79; // rsi
  int v80; // r8d
  int v81; // r8d
  __int64 v82; // r9
  __int64 *v83; // rdi
  unsigned int v84; // ebx
  int v85; // ecx
  __int64 v86; // rsi
  int v87; // r10d
  __int64 v88; // [rsp+0h] [rbp-C0h]
  __int64 v90; // [rsp+10h] [rbp-B0h]
  __int64 v91; // [rsp+18h] [rbp-A8h]
  __int64 v92; // [rsp+20h] [rbp-A0h]
  __int64 v93; // [rsp+28h] [rbp-98h]
  __int64 *v94; // [rsp+30h] [rbp-90h]
  __int64 v96; // [rsp+40h] [rbp-80h]
  _BYTE v97[112]; // [rsp+50h] [rbp-70h] BYREF

  v91 = *(_QWORD *)(a1 + 168);
  if ( a2 )
  {
    v2 = (__int64 **)(a1 + 680);
  }
  else
  {
    v91 = *(_QWORD *)(a1 + 176);
    v2 = (__int64 **)(a1 + 536);
  }
  v3 = v97;
  sub_1B3B830(v97, 0);
  v94 = *v2;
  v90 = (__int64)&(*v2)[*((unsigned int *)v2 + 2)];
  v88 = a1 + 648;
  if ( *v2 != (__int64 *)v90 )
  {
    do
    {
      v4 = *(_QWORD *)(*v94 - 24);
      v93 = *v94;
      v5 = *(_QWORD *)(*v94 - 48);
      v96 = *(_QWORD *)(*v94 + 40);
      sub_1B3B8C0(v3, *(_QWORD *)(a1 + 160), byte_3F871B3, 0);
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 80LL);
      if ( v6 )
        v6 -= 24;
      sub_1B3BE00(v3, v6, v91);
      if ( a2 )
      {
        sub_1B3BE00(v3, v5, v91);
        v7 = *(_DWORD *)(a1 + 672);
        if ( !v7 )
        {
          ++*(_QWORD *)(a1 + 648);
          goto LABEL_107;
        }
        v8 = *(_QWORD *)(a1 + 656);
        v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v10 = (__int64 *)(v8 + 40LL * v9);
        v11 = *v10;
        if ( v5 != *v10 )
        {
          v56 = 1;
          v57 = 0;
          while ( v11 != -8 )
          {
            if ( v11 == -16 && !v57 )
              v57 = v10;
            v9 = (v7 - 1) & (v56 + v9);
            v10 = (__int64 *)(v8 + 40LL * v9);
            v11 = *v10;
            if ( v5 == *v10 )
              goto LABEL_9;
            ++v56;
          }
          if ( v57 )
            v10 = v57;
          ++*(_QWORD *)(a1 + 648);
          v58 = *(_DWORD *)(a1 + 664) + 1;
          if ( 4 * v58 < 3 * v7 )
          {
            if ( v7 - *(_DWORD *)(a1 + 668) - v58 <= v7 >> 3 )
            {
              sub_1A6F390(v88, v7);
              v73 = *(_DWORD *)(a1 + 672);
              if ( !v73 )
              {
LABEL_158:
                ++*(_DWORD *)(a1 + 664);
                BUG();
              }
              v74 = v73 - 1;
              v75 = *(_QWORD *)(a1 + 656);
              v76 = 0;
              v77 = v74 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
              v58 = *(_DWORD *)(a1 + 664) + 1;
              v78 = 1;
              v10 = (__int64 *)(v75 + 40LL * v77);
              v79 = *v10;
              if ( v5 != *v10 )
              {
                while ( v79 != -8 )
                {
                  if ( !v76 && v79 == -16 )
                    v76 = v10;
                  v77 = v74 & (v78 + v77);
                  v10 = (__int64 *)(v75 + 40LL * v77);
                  v79 = *v10;
                  if ( v5 == *v10 )
                    goto LABEL_95;
                  ++v78;
                }
                if ( v76 )
                  v10 = v76;
              }
            }
            goto LABEL_95;
          }
LABEL_107:
          sub_1A6F390(v88, 2 * v7);
          v66 = *(_DWORD *)(a1 + 672);
          if ( !v66 )
            goto LABEL_158;
          v67 = v66 - 1;
          v68 = *(_QWORD *)(a1 + 656);
          v69 = v67 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v58 = *(_DWORD *)(a1 + 664) + 1;
          v10 = (__int64 *)(v68 + 40LL * v69);
          v70 = *v10;
          if ( v5 != *v10 )
          {
            v71 = 1;
            v72 = 0;
            while ( v70 != -8 )
            {
              if ( !v72 && v70 == -16 )
                v72 = v10;
              v69 = v67 & (v71 + v69);
              v10 = (__int64 *)(v68 + 40LL * v69);
              v70 = *v10;
              if ( v5 == *v10 )
                goto LABEL_95;
              ++v71;
            }
            if ( v72 )
              v10 = v72;
          }
LABEL_95:
          *(_DWORD *)(a1 + 664) = v58;
          if ( *v10 != -8 )
            --*(_DWORD *)(a1 + 668);
          *v10 = v5;
          v10[1] = 0;
          v10[2] = 0;
          v10[3] = 0;
          *((_DWORD *)v10 + 8) = 0;
          goto LABEL_14;
        }
      }
      else
      {
        sub_1B3BE00(v3, v96, v91);
        v45 = *(_DWORD *)(a1 + 528);
        v46 = a1 + 504;
        if ( !v45 )
        {
          ++*(_QWORD *)(a1 + 504);
          goto LABEL_99;
        }
        v47 = *(_QWORD *)(a1 + 512);
        LODWORD(v48) = (v45 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v10 = (__int64 *)(v47 + 40LL * (unsigned int)v48);
        v49 = *v10;
        if ( v4 != *v10 )
        {
          v50 = 1;
          v51 = 0;
          while ( v49 != -8 )
          {
            if ( !v51 && v49 == -16 )
              v51 = v10;
            v48 = (v45 - 1) & ((_DWORD)v48 + v50);
            v10 = (__int64 *)(v47 + 40 * v48);
            v49 = *v10;
            if ( v4 == *v10 )
              goto LABEL_9;
            ++v50;
          }
          if ( v51 )
            v10 = v51;
          ++*(_QWORD *)(a1 + 504);
          v52 = *(_DWORD *)(a1 + 520) + 1;
          if ( 4 * v52 < 3 * v45 )
          {
            if ( v45 - *(_DWORD *)(a1 + 524) - v52 <= v45 >> 3 )
            {
              sub_1A6F390(v46, v45);
              v80 = *(_DWORD *)(a1 + 528);
              if ( !v80 )
              {
LABEL_159:
                ++*(_DWORD *)(a1 + 520);
                BUG();
              }
              v81 = v80 - 1;
              v82 = *(_QWORD *)(a1 + 512);
              v83 = 0;
              v84 = v81 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
              v52 = *(_DWORD *)(a1 + 520) + 1;
              v85 = 1;
              v10 = (__int64 *)(v82 + 40LL * v84);
              v86 = *v10;
              if ( v4 != *v10 )
              {
                while ( v86 != -8 )
                {
                  if ( v86 == -16 && !v83 )
                    v83 = v10;
                  v84 = v81 & (v85 + v84);
                  v10 = (__int64 *)(v82 + 40LL * v84);
                  v86 = *v10;
                  if ( v4 == *v10 )
                    goto LABEL_77;
                  ++v85;
                }
                if ( v83 )
                  v10 = v83;
              }
            }
LABEL_77:
            *(_DWORD *)(a1 + 520) = v52;
            if ( *v10 != -8 )
              --*(_DWORD *)(a1 + 524);
            *v10 = v4;
            v10[1] = 0;
            v10[2] = 0;
            v10[3] = 0;
            *((_DWORD *)v10 + 8) = 0;
LABEL_14:
            v17 = v96;
LABEL_15:
            sub_1B3BE00(v3, v17, v91);
            goto LABEL_16;
          }
LABEL_99:
          sub_1A6F390(v46, 2 * v45);
          v59 = *(_DWORD *)(a1 + 528);
          if ( !v59 )
            goto LABEL_159;
          v60 = v59 - 1;
          v61 = *(_QWORD *)(a1 + 512);
          v62 = v60 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v52 = *(_DWORD *)(a1 + 520) + 1;
          v10 = (__int64 *)(v61 + 40LL * v62);
          v63 = *v10;
          if ( v4 != *v10 )
          {
            v64 = 1;
            v65 = 0;
            while ( v63 != -8 )
            {
              if ( v63 == -16 && !v65 )
                v65 = v10;
              v62 = v60 & (v64 + v62);
              v10 = (__int64 *)(v61 + 40LL * v62);
              v63 = *v10;
              if ( v4 == *v10 )
                goto LABEL_77;
              ++v64;
            }
            if ( v65 )
              v10 = v65;
          }
          goto LABEL_77;
        }
      }
LABEL_9:
      v12 = v10 + 1;
      v92 = *(_QWORD *)(a1 + 216);
      if ( !*((_DWORD *)v10 + 6) )
        goto LABEL_14;
      v13 = (__int64 *)v10[2];
      v14 = &v13[2 * *((unsigned int *)v12 + 6)];
      if ( v13 == v14 )
        goto LABEL_14;
      while ( 1 )
      {
        v15 = *v13;
        v16 = v13;
        if ( *v13 != -16 && v15 != -8 )
          break;
        v13 += 2;
        if ( v14 == v13 )
          goto LABEL_14;
      }
      if ( v13 == v14 )
        goto LABEL_14;
      v23 = v3;
      v17 = v96;
      v24 = v13[1];
      v25 = 0;
      if ( v96 != v15 )
      {
        while ( 1 )
        {
          sub_1B3BE00(v23, v15, v24);
          if ( v17 )
            break;
          v17 = v15;
          v25 = 1;
LABEL_47:
          v16 += 2;
          if ( v16 == v14 )
            goto LABEL_51;
          while ( 1 )
          {
            v15 = *v16;
            if ( *v16 != -16 && v15 != -8 )
              break;
            v16 += 2;
            if ( v14 == v16 )
              goto LABEL_51;
          }
          if ( v14 == v16 )
          {
LABEL_51:
            v41 = v25;
            v3 = v23;
            goto LABEL_52;
          }
          v24 = v16[1];
          if ( v96 == v15 )
            goto LABEL_56;
        }
        v26 = *(_QWORD *)(*(_QWORD *)(v17 + 56) + 80LL);
        if ( v26 )
        {
          v27 = v26 - 24;
          if ( v17 == v27 )
            goto LABEL_45;
          if ( v15 != v27 )
            goto LABEL_31;
LABEL_64:
          if ( v17 == v27 )
          {
LABEL_45:
            if ( v15 == v17 )
              v25 = 1;
            goto LABEL_47;
          }
LABEL_44:
          v17 = v27;
          v25 = 0;
          goto LABEL_45;
        }
        if ( !v15 )
          goto LABEL_43;
LABEL_31:
        v28 = *(_QWORD *)(v92 + 32);
        v29 = *(unsigned int *)(v92 + 48);
        if ( !(_DWORD)v29 )
          goto LABEL_43;
        v30 = v29 - 1;
        v31 = (v29 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v32 = (__int64 *)(v28 + 16LL * v31);
        v33 = *v32;
        if ( v17 == *v32 )
        {
LABEL_33:
          v34 = (__int64 *)(v28 + 16 * v29);
          if ( v32 != v34 )
          {
            v35 = (__int64 *)v32[1];
LABEL_35:
            v36 = v30 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v37 = (__int64 *)(v28 + 16LL * v36);
            v38 = *v37;
            if ( v15 == *v37 )
            {
LABEL_36:
              if ( v37 != v34 )
              {
                v39 = (__int64 *)v37[1];
                if ( v35 )
                {
                  if ( v39 )
                  {
                    while ( v35 != v39 )
                    {
                      if ( *((_DWORD *)v35 + 4) < *((_DWORD *)v39 + 4) )
                      {
                        v40 = v35;
                        v35 = v39;
                        v39 = v40;
                      }
                      v35 = (__int64 *)v35[1];
                      if ( !v35 )
                        goto LABEL_43;
                    }
                    v27 = *v35;
                    goto LABEL_64;
                  }
                }
              }
            }
            else
            {
              v53 = 1;
              while ( v38 != -8 )
              {
                v54 = v53 + 1;
                v36 = v30 & (v53 + v36);
                v37 = (__int64 *)(v28 + 16LL * v36);
                v38 = *v37;
                if ( v15 == *v37 )
                  goto LABEL_36;
                v53 = v54;
              }
            }
LABEL_43:
            v27 = 0;
            goto LABEL_44;
          }
        }
        else
        {
          v55 = 1;
          while ( v33 != -8 )
          {
            v87 = v55 + 1;
            v31 = v30 & (v55 + v31);
            v32 = (__int64 *)(v28 + 16LL * v31);
            v33 = *v32;
            if ( v17 == *v32 )
              goto LABEL_33;
            v55 = v87;
          }
          v34 = (__int64 *)(v28 + 16LL * (unsigned int)v29);
        }
        v35 = 0;
        goto LABEL_35;
      }
LABEL_56:
      v41 = v25;
      v3 = v23;
      if ( v24 )
      {
        if ( *(_QWORD *)(v93 - 72) )
        {
          v42 = *(_QWORD *)(v93 - 64);
          v43 = *(_QWORD *)(v93 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v43 = v42;
          if ( v42 )
            *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
        }
        *(_QWORD *)(v93 - 72) = v24;
        v44 = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v93 - 64) = v44;
        if ( v44 )
          *(_QWORD *)(v44 + 16) = (v93 - 64) | *(_QWORD *)(v44 + 16) & 3LL;
        *(_QWORD *)(v93 - 56) = *(_QWORD *)(v93 - 56) & 3LL | (v24 + 8);
        *(_QWORD *)(v24 + 8) = v93 - 72;
        goto LABEL_23;
      }
LABEL_52:
      if ( !v41 )
        goto LABEL_15;
LABEL_16:
      v18 = sub_1B40B40(v3);
      if ( *(_QWORD *)(v93 - 72) )
      {
        v19 = *(_QWORD *)(v93 - 64);
        v20 = *(_QWORD *)(v93 - 56) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v20 = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
      }
      *(_QWORD *)(v93 - 72) = v18;
      if ( v18 )
      {
        v21 = *(_QWORD *)(v18 + 8);
        *(_QWORD *)(v93 - 64) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = (v93 - 64) | *(_QWORD *)(v21 + 16) & 3LL;
        *(_QWORD *)(v93 - 56) = *(_QWORD *)(v93 - 56) & 3LL | (v18 + 8);
        *(_QWORD *)(v18 + 8) = v93 - 72;
      }
LABEL_23:
      ++v94;
    }
    while ( (__int64 *)v90 != v94 );
  }
  return sub_1B3B860(v3);
}
