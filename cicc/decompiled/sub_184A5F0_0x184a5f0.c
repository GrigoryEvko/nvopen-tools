// Function: sub_184A5F0
// Address: 0x184a5f0
//
__int64 *__fastcall sub_184A5F0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v3; // rax
  __int64 *result; // rax
  __int64 v5; // r13
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  unsigned int v10; // esi
  __int64 *v11; // rdx
  __int64 v12; // r9
  unsigned int v13; // edx
  int v14; // edx
  unsigned int v15; // r12d
  __int64 *v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // r8d
  __int64 *v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  _BYTE *v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned int v26; // edx
  __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // rdx
  _BYTE *v30; // r12
  unsigned int v31; // esi
  int v32; // esi
  __int64 v33; // rdi
  int v34; // esi
  __int64 v35; // r9
  int v36; // edx
  unsigned int v37; // ecx
  __int64 v38; // r8
  __int64 v39; // rdx
  int v40; // r11d
  __int64 *v41; // r10
  int v42; // edi
  int v43; // ecx
  __int64 v44; // rdi
  int v45; // ecx
  __int64 v46; // r9
  __int64 *v47; // r10
  int v48; // r11d
  unsigned int v49; // esi
  __int64 v50; // r8
  int v51; // r10d
  int v52; // r11d
  __int64 *v53; // rcx
  int v54; // eax
  int v55; // edi
  int v56; // eax
  int v57; // esi
  __int64 v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // r8
  int v61; // r10d
  __int64 *v62; // r9
  int v63; // eax
  int v64; // eax
  __int64 v65; // r8
  int v66; // r10d
  unsigned int v67; // edx
  __int64 v68; // rsi
  int v69; // r11d
  int v70; // ecx
  __int64 *v71; // r11
  unsigned int v72; // [rsp+8h] [rbp-38h]

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  while ( 1 )
  {
    result = *(__int64 **)(a1 + 96);
    if ( *(__int64 **)(a1 + 88) == result )
      return result;
    while ( 1 )
    {
      v5 = *(result - 3);
      v6 = (__int64 *)*(result - 2);
      if ( v6 == (__int64 *)(*(_QWORD *)(v5 + 8) + 8LL * *(unsigned int *)(v5 + 16)) )
        break;
      *(result - 2) = (__int64)(v6 + 1);
      v7 = *(unsigned int *)(a1 + 32);
      v8 = *v6;
      if ( !(_DWORD)v7 )
        goto LABEL_12;
      v9 = *(_QWORD *)(a1 + 16);
      v10 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v8 == *v11 )
      {
LABEL_7:
        if ( v11 == (__int64 *)(v9 + 16 * v7) )
          goto LABEL_12;
        result = *(__int64 **)(a1 + 96);
        v13 = *((_DWORD *)v11 + 2);
        if ( v13 < *((_DWORD *)result - 2) )
        {
          *((_DWORD *)result - 2) = v13;
          result = *(__int64 **)(a1 + 96);
        }
      }
      else
      {
        v14 = 1;
        while ( v12 != -8 )
        {
          v51 = v14 + 1;
          v10 = (v7 - 1) & (v14 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v8 == *v11 )
            goto LABEL_7;
          v14 = v51;
        }
LABEL_12:
        sub_184A1D0((int *)a1, v8);
        result = *(__int64 **)(a1 + 96);
      }
    }
    v15 = *((_DWORD *)result - 2);
    v16 = result - 3;
    *(_QWORD *)(a1 + 96) = v16;
    if ( *(__int64 **)(a1 + 88) != v16 && *((_DWORD *)v16 - 2) > v15 )
      *((_DWORD *)v16 - 2) = v15;
    v17 = *(_DWORD *)(a1 + 32);
    if ( !v17 )
    {
      ++*(_QWORD *)(a1 + 8);
LABEL_65:
      sub_1849300(a1 + 8, 2 * v17);
      v56 = *(_DWORD *)(a1 + 32);
      if ( v56 )
      {
        v57 = v56 - 1;
        v58 = *(_QWORD *)(a1 + 16);
        v59 = (v56 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v55 = *(_DWORD *)(a1 + 24) + 1;
        v53 = (__int64 *)(v58 + 16LL * v59);
        v60 = *v53;
        if ( *v53 == v5 )
          goto LABEL_61;
        v61 = 1;
        v62 = 0;
        while ( v60 != -8 )
        {
          if ( !v62 && v60 == -16 )
            v62 = v53;
          v59 = v57 & (v61 + v59);
          v53 = (__int64 *)(v58 + 16LL * v59);
          v60 = *v53;
          if ( *v53 == v5 )
            goto LABEL_61;
          ++v61;
        }
LABEL_69:
        if ( v62 )
          v53 = v62;
        goto LABEL_61;
      }
LABEL_104:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
    v18 = *(_QWORD *)(a1 + 16);
    v19 = (v17 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v20 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v20;
    if ( *v20 == v5 )
    {
      v22 = *((_DWORD *)v20 + 2);
      goto LABEL_19;
    }
    v52 = 1;
    v53 = 0;
    while ( v21 != -8 )
    {
      if ( v53 || v21 != -16 )
        v20 = v53;
      v70 = v52 + 1;
      v19 = (v17 - 1) & (v52 + v19);
      v71 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v71;
      if ( *v71 == v5 )
      {
        v22 = *((_DWORD *)v71 + 2);
        goto LABEL_19;
      }
      v52 = v70;
      v53 = v20;
      v20 = (__int64 *)(v18 + 16LL * v19);
    }
    if ( !v53 )
      v53 = v20;
    v54 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v55 = v54 + 1;
    if ( 4 * (v54 + 1) >= 3 * v17 )
      goto LABEL_65;
    if ( v17 - *(_DWORD *)(a1 + 28) - v55 <= v17 >> 3 )
    {
      v72 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
      sub_1849300(a1 + 8, v17);
      v63 = *(_DWORD *)(a1 + 32);
      if ( v63 )
      {
        v64 = v63 - 1;
        v65 = *(_QWORD *)(a1 + 16);
        v66 = 1;
        v62 = 0;
        v67 = v64 & v72;
        v55 = *(_DWORD *)(a1 + 24) + 1;
        v53 = (__int64 *)(v65 + 16LL * (v64 & v72));
        v68 = *v53;
        if ( *v53 == v5 )
          goto LABEL_61;
        while ( v68 != -8 )
        {
          if ( !v62 && v68 == -16 )
            v62 = v53;
          v67 = v64 & (v66 + v67);
          v53 = (__int64 *)(v65 + 16LL * v67);
          v68 = *v53;
          if ( *v53 == v5 )
            goto LABEL_61;
          ++v66;
        }
        goto LABEL_69;
      }
      goto LABEL_104;
    }
LABEL_61:
    *(_DWORD *)(a1 + 24) = v55;
    if ( *v53 != -8 )
      --*(_DWORD *)(a1 + 28);
    *v53 = v5;
    v22 = 0;
    *((_DWORD *)v53 + 2) = 0;
LABEL_19:
    if ( v15 == v22 )
    {
      v23 = *(_BYTE **)(a1 + 72);
      while ( 1 )
      {
        v28 = *(_QWORD *)(a1 + 48);
        v29 = (_QWORD *)(v28 - 8);
        if ( *(_BYTE **)(a1 + 80) == v23 )
        {
          sub_18483E0(v1, v23, v29);
          v30 = *(_BYTE **)(a1 + 72);
          v29 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
        }
        else
        {
          if ( v23 )
          {
            *(_QWORD *)v23 = *(_QWORD *)(v28 - 8);
            v23 = *(_BYTE **)(a1 + 72);
            v29 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          v30 = v23 + 8;
          *(_QWORD *)(a1 + 72) = v23 + 8;
        }
        v31 = *(_DWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 48) = v29;
        if ( !v31 )
          break;
        v24 = *((_QWORD *)v30 - 1);
        v25 = *(_QWORD *)(a1 + 16);
        v26 = (v31 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        result = (__int64 *)(v25 + 16LL * v26);
        v27 = *result;
        if ( *result != v24 )
        {
          v40 = 1;
          v41 = 0;
          while ( v27 != -8 )
          {
            if ( !v41 && v27 == -16 )
              v41 = result;
            v26 = (v31 - 1) & (v40 + v26);
            result = (__int64 *)(v25 + 16LL * v26);
            v27 = *result;
            if ( v24 == *result )
              goto LABEL_22;
            ++v40;
          }
          v42 = *(_DWORD *)(a1 + 24);
          if ( v41 )
            result = v41;
          ++*(_QWORD *)(a1 + 8);
          v36 = v42 + 1;
          if ( 4 * (v42 + 1) < 3 * v31 )
          {
            if ( v31 - *(_DWORD *)(a1 + 28) - v36 <= v31 >> 3 )
            {
              sub_1849300(a1 + 8, v31);
              v43 = *(_DWORD *)(a1 + 32);
              if ( !v43 )
              {
LABEL_103:
                ++*(_DWORD *)(a1 + 24);
                BUG();
              }
              v44 = *((_QWORD *)v30 - 1);
              v45 = v43 - 1;
              v46 = *(_QWORD *)(a1 + 16);
              v47 = 0;
              v48 = 1;
              v36 = *(_DWORD *)(a1 + 24) + 1;
              v49 = v45 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
              result = (__int64 *)(v46 + 16LL * v49);
              v50 = *result;
              if ( *result != v44 )
              {
                while ( v50 != -8 )
                {
                  if ( !v47 && v50 == -16 )
                    v47 = result;
                  v49 = v45 & (v48 + v49);
                  result = (__int64 *)(v46 + 16LL * v49);
                  v50 = *result;
                  if ( v44 == *result )
                    goto LABEL_31;
                  ++v48;
                }
LABEL_45:
                if ( v47 )
                  result = v47;
              }
            }
LABEL_31:
            *(_DWORD *)(a1 + 24) = v36;
            if ( *result != -8 )
              --*(_DWORD *)(a1 + 28);
            v39 = *((_QWORD *)v30 - 1);
            *((_DWORD *)result + 2) = 0;
            *result = v39;
            goto LABEL_22;
          }
LABEL_29:
          sub_1849300(a1 + 8, 2 * v31);
          v32 = *(_DWORD *)(a1 + 32);
          if ( !v32 )
            goto LABEL_103;
          v33 = *((_QWORD *)v30 - 1);
          v34 = v32 - 1;
          v35 = *(_QWORD *)(a1 + 16);
          v36 = *(_DWORD *)(a1 + 24) + 1;
          v37 = v34 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          result = (__int64 *)(v35 + 16LL * v37);
          v38 = *result;
          if ( *result != v33 )
          {
            v69 = 1;
            v47 = 0;
            while ( v38 != -8 )
            {
              if ( v38 == -16 && !v47 )
                v47 = result;
              v37 = v34 & (v69 + v37);
              result = (__int64 *)(v35 + 16LL * v37);
              v38 = *result;
              if ( v33 == *result )
                goto LABEL_31;
              ++v69;
            }
            goto LABEL_45;
          }
          goto LABEL_31;
        }
LABEL_22:
        *((_DWORD *)result + 2) = -1;
        v23 = *(_BYTE **)(a1 + 72);
        if ( *((_QWORD *)v23 - 1) == v5 )
          return result;
      }
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_29;
    }
  }
}
