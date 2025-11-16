// Function: sub_2623260
// Address: 0x2623260
//
void __fastcall sub_2623260(char **a1, __int64 a2)
{
  char *v3; // rsi
  char *v4; // r12
  unsigned __int64 v5; // rax
  char *v6; // rbx
  __int64 v7; // r14
  unsigned int v8; // r10d
  unsigned int v9; // ecx
  __int64 v10; // rax
  int v11; // r13d
  unsigned int v12; // r9d
  __int64 *v13; // rdi
  __int64 *v14; // rdx
  __int64 v15; // r8
  unsigned int v16; // edi
  unsigned int v17; // r13d
  unsigned int v18; // r9d
  __int64 *v19; // rdx
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 v22; // r12
  char *v23; // r11
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // r8
  unsigned int v27; // edi
  __int64 v28; // rsi
  int v29; // eax
  int v30; // esi
  int v31; // esi
  __int64 v32; // r8
  __int64 v33; // rcx
  int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // rdi
  int v37; // eax
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // r8
  __int64 *v41; // r9
  int v42; // r13d
  unsigned int v43; // edi
  __int64 v44; // rsi
  int v45; // ebx
  int v46; // ecx
  int v47; // ecx
  __int64 v48; // rdi
  __int64 *v49; // r8
  __int64 v50; // r13
  int v51; // r9d
  __int64 v52; // rsi
  int v53; // r10d
  __int64 *v54; // r9
  int v55; // r13d
  __int64 v56; // [rsp+8h] [rbp-58h]
  int v57; // [rsp+10h] [rbp-50h]
  unsigned int v58; // [rsp+18h] [rbp-48h]
  char *v59; // [rsp+18h] [rbp-48h]
  unsigned int v60; // [rsp+18h] [rbp-48h]
  __int64 *v61; // [rsp+18h] [rbp-48h]
  char *v62; // [rsp+18h] [rbp-48h]
  char *v63; // [rsp+20h] [rbp-40h]
  char *v64; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v63 = v3;
  if ( *a1 != v3 )
  {
    _BitScanReverse64(&v5, (v3 - v4) >> 3);
    sub_2621290(v4, v3, 2LL * (int)(63 - (v5 ^ 0x3F)), a2);
    if ( v3 - v4 <= 128 )
    {
      sub_2622810(v4, v3, a2);
      return;
    }
    sub_2622810(v4, v4 + 128, a2);
    v64 = v4 + 128;
    if ( v3 != v4 + 128 )
    {
      do
      {
        v6 = v64;
        v7 = *(_QWORD *)v64;
        v8 = ((unsigned int)*(_QWORD *)v64 >> 9) ^ ((unsigned int)*(_QWORD *)v64 >> 4);
        while ( 1 )
        {
          v21 = *(_DWORD *)(a2 + 24);
          v22 = *((_QWORD *)v6 - 1);
          v23 = v6;
          if ( v21 )
          {
            v9 = v21 - 1;
            v10 = *(_QWORD *)(a2 + 8);
            v11 = 1;
            v12 = (v21 - 1) & v8;
            v13 = (__int64 *)(v10 + 40LL * v12);
            v14 = 0;
            v15 = *v13;
            if ( v7 == *v13 )
            {
LABEL_6:
              v16 = *((_DWORD *)v13 + 2);
              goto LABEL_7;
            }
            while ( v15 != -4096 )
            {
              if ( !v14 && v15 == -8192 )
                v14 = v13;
              v12 = v9 & (v11 + v12);
              v13 = (__int64 *)(v10 + 40LL * v12);
              v15 = *v13;
              if ( v7 == *v13 )
                goto LABEL_6;
              ++v11;
            }
            v37 = *(_DWORD *)(a2 + 16);
            if ( !v14 )
              v14 = v13;
            ++*(_QWORD *)a2;
            v29 = v37 + 1;
            if ( 4 * v29 < 3 * v21 )
            {
              if ( v21 - *(_DWORD *)(a2 + 20) - v29 > v21 >> 3 )
                goto LABEL_14;
              v60 = v8;
              sub_261D190(a2, v21);
              v38 = *(_DWORD *)(a2 + 24);
              if ( !v38 )
                goto LABEL_88;
              v8 = v60;
              v39 = v38 - 1;
              v40 = *(_QWORD *)(a2 + 8);
              v41 = 0;
              v23 = v6;
              v42 = 1;
              v43 = v39 & v60;
              v14 = (__int64 *)(v40 + 40LL * (v39 & v60));
              v44 = *v14;
              v29 = *(_DWORD *)(a2 + 16) + 1;
              if ( v7 == *v14 )
                goto LABEL_14;
              while ( v44 != -4096 )
              {
                if ( v44 == -8192 && !v41 )
                  v41 = v14;
                v43 = v39 & (v42 + v43);
                v14 = (__int64 *)(v40 + 40LL * v43);
                v44 = *v14;
                if ( v7 == *v14 )
                  goto LABEL_14;
                ++v42;
              }
              goto LABEL_39;
            }
          }
          else
          {
            ++*(_QWORD *)a2;
          }
          v58 = v8;
          sub_261D190(a2, 2 * v21);
          v24 = *(_DWORD *)(a2 + 24);
          if ( !v24 )
            goto LABEL_88;
          v8 = v58;
          v25 = v24 - 1;
          v26 = *(_QWORD *)(a2 + 8);
          v23 = v6;
          v27 = v25 & v58;
          v14 = (__int64 *)(v26 + 40LL * (v25 & v58));
          v28 = *v14;
          v29 = *(_DWORD *)(a2 + 16) + 1;
          if ( v7 == *v14 )
            goto LABEL_14;
          v55 = 1;
          v41 = 0;
          while ( v28 != -4096 )
          {
            if ( !v41 && v28 == -8192 )
              v41 = v14;
            v27 = v25 & (v55 + v27);
            v14 = (__int64 *)(v26 + 40LL * v27);
            v28 = *v14;
            if ( v7 == *v14 )
              goto LABEL_14;
            ++v55;
          }
LABEL_39:
          if ( v41 )
            v14 = v41;
LABEL_14:
          *(_DWORD *)(a2 + 16) = v29;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a2 + 20);
          *v14 = v7;
          *(_OWORD *)(v14 + 1) = 0;
          *(_OWORD *)(v14 + 3) = 0;
          v21 = *(_DWORD *)(a2 + 24);
          if ( !v21 )
          {
            ++*(_QWORD *)a2;
            goto LABEL_18;
          }
          v10 = *(_QWORD *)(a2 + 8);
          v9 = v21 - 1;
          v16 = 0;
LABEL_7:
          v17 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
          v18 = v17 & v9;
          v19 = (__int64 *)(v10 + 40LL * (v17 & v9));
          v20 = *v19;
          if ( v22 != *v19 )
            break;
LABEL_8:
          v6 -= 8;
          if ( v16 >= *((_DWORD *)v19 + 2) )
            goto LABEL_23;
          *((_QWORD *)v6 + 1) = *(_QWORD *)v6;
        }
        v57 = 1;
        v61 = 0;
        v56 = v10;
        while ( 1 )
        {
          v35 = v61;
          if ( v20 == -4096 )
            break;
          if ( !v61 )
          {
            if ( v20 != -8192 )
              v19 = 0;
            v61 = v19;
          }
          v18 = v9 & (v57 + v18);
          v19 = (__int64 *)(v56 + 40LL * v18);
          v20 = *v19;
          if ( v22 == *v19 )
            goto LABEL_8;
          ++v57;
        }
        v45 = *(_DWORD *)(a2 + 16);
        if ( !v61 )
          v35 = v19;
        ++*(_QWORD *)a2;
        v34 = v45 + 1;
        if ( 4 * (v45 + 1) >= 3 * v21 )
        {
LABEL_18:
          v59 = v23;
          sub_261D190(a2, 2 * v21);
          v30 = *(_DWORD *)(a2 + 24);
          if ( !v30 )
            goto LABEL_88;
          v31 = v30 - 1;
          v32 = *(_QWORD *)(a2 + 8);
          v23 = v59;
          LODWORD(v33) = v31 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v34 = *(_DWORD *)(a2 + 16) + 1;
          v35 = (__int64 *)(v32 + 40LL * (unsigned int)v33);
          v36 = *v35;
          if ( v22 != *v35 )
          {
            v53 = 1;
            v54 = 0;
            while ( v36 != -4096 )
            {
              if ( !v54 && v36 == -8192 )
                v54 = v35;
              v33 = v31 & (unsigned int)(v33 + v53);
              v35 = (__int64 *)(v32 + 40 * v33);
              v36 = *v35;
              if ( v22 == *v35 )
                goto LABEL_20;
              ++v53;
            }
            if ( v54 )
              v35 = v54;
          }
        }
        else if ( v21 - (v34 + *(_DWORD *)(a2 + 20)) <= v21 >> 3 )
        {
          v62 = v23;
          sub_261D190(a2, v21);
          v46 = *(_DWORD *)(a2 + 24);
          if ( v46 )
          {
            v47 = v46 - 1;
            v48 = *(_QWORD *)(a2 + 8);
            v49 = 0;
            LODWORD(v50) = v47 & v17;
            v23 = v62;
            v51 = 1;
            v34 = *(_DWORD *)(a2 + 16) + 1;
            v35 = (__int64 *)(v48 + 40LL * (unsigned int)v50);
            v52 = *v35;
            if ( v22 != *v35 )
            {
              while ( v52 != -4096 )
              {
                if ( !v49 && v52 == -8192 )
                  v49 = v35;
                v50 = v47 & (unsigned int)(v50 + v51);
                v35 = (__int64 *)(v48 + 40 * v50);
                v52 = *v35;
                if ( v22 == *v35 )
                  goto LABEL_20;
                ++v51;
              }
              if ( v49 )
                v35 = v49;
            }
            goto LABEL_20;
          }
LABEL_88:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
LABEL_20:
        *(_DWORD *)(a2 + 16) = v34;
        if ( *v35 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v35 = v22;
        *(_OWORD *)(v35 + 1) = 0;
        *(_OWORD *)(v35 + 3) = 0;
LABEL_23:
        v64 += 8;
        *(_QWORD *)v23 = v7;
      }
      while ( v63 != v64 );
    }
  }
}
