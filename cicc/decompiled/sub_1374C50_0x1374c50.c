// Function: sub_1374C50
// Address: 0x1374c50
//
__int64 *__fastcall sub_1374C50(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 *result; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // r12d
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r9
  unsigned int v11; // r8d
  __int64 *v12; // rax
  __int64 v13; // rdi
  int v14; // eax
  _BYTE *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rdx
  _BYTE *v22; // r12
  unsigned int v23; // esi
  int v24; // esi
  __int64 v25; // rdi
  int v26; // esi
  __int64 v27; // r9
  int v28; // edx
  unsigned int v29; // ecx
  __int64 v30; // r8
  __int64 v31; // rdx
  int v32; // r11d
  __int64 *v33; // r10
  int v34; // edi
  int v35; // esi
  __int64 v36; // rdi
  int v37; // esi
  __int64 v38; // r9
  __int64 *v39; // r10
  int v40; // r11d
  unsigned int v41; // ecx
  __int64 v42; // r8
  int v43; // r11d
  __int64 *v44; // rcx
  int v45; // eax
  int v46; // edi
  int v47; // eax
  int v48; // edx
  __int64 v49; // rsi
  unsigned int v50; // eax
  __int64 v51; // r8
  int v52; // r10d
  __int64 *v53; // r9
  int v54; // eax
  int v55; // eax
  __int64 v56; // r8
  int v57; // r10d
  unsigned int v58; // edx
  __int64 v59; // rsi
  int v60; // r11d
  int v61; // ecx
  __int64 *v62; // r11
  unsigned int v63; // [rsp+8h] [rbp-38h]

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  while ( 1 )
  {
    result = *(__int64 **)(a1 + 96);
    if ( *(__int64 **)(a1 + 88) == result )
      return result;
    sub_1374B30(a1);
    v5 = *(_QWORD *)(a1 + 96);
    v6 = *(_QWORD *)(v5 - 48);
    v7 = *(_DWORD *)(v5 - 8);
    v8 = v5 - 48;
    *(_QWORD *)(a1 + 96) = v8;
    if ( *(_QWORD *)(a1 + 88) != v8 && *(_DWORD *)(v8 - 8) > v7 )
      *(_DWORD *)(v8 - 8) = v7;
    v9 = *(_DWORD *)(a1 + 32);
    if ( v9 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
      {
        v14 = *((_DWORD *)v12 + 2);
        goto LABEL_10;
      }
      v43 = 1;
      v44 = 0;
      while ( v13 != -8 )
      {
        if ( v44 || v13 != -16 )
          v12 = v44;
        v61 = v43 + 1;
        v11 = (v9 - 1) & (v43 + v11);
        v62 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v62;
        if ( v6 == *v62 )
        {
          v14 = *((_DWORD *)v62 + 2);
          goto LABEL_10;
        }
        v43 = v61;
        v44 = v12;
        v12 = (__int64 *)(v10 + 16LL * v11);
      }
      if ( !v44 )
        v44 = v12;
      v45 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v46 = v45 + 1;
      if ( 4 * (v45 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 28) - v46 > v9 >> 3 )
          goto LABEL_50;
        v63 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
        sub_13745E0(a1 + 8, v9);
        v54 = *(_DWORD *)(a1 + 32);
        if ( !v54 )
        {
LABEL_92:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 16);
        v57 = 1;
        v53 = 0;
        v58 = v55 & v63;
        v46 = *(_DWORD *)(a1 + 24) + 1;
        v44 = (__int64 *)(v56 + 16LL * (v55 & v63));
        v59 = *v44;
        if ( v6 == *v44 )
          goto LABEL_50;
        while ( v59 != -8 )
        {
          if ( v59 == -16 && !v53 )
            v53 = v44;
          v58 = v55 & (v57 + v58);
          v44 = (__int64 *)(v56 + 16LL * v58);
          v59 = *v44;
          if ( v6 == *v44 )
            goto LABEL_50;
          ++v57;
        }
        goto LABEL_58;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    sub_13745E0(a1 + 8, 2 * v9);
    v47 = *(_DWORD *)(a1 + 32);
    if ( !v47 )
      goto LABEL_92;
    v48 = v47 - 1;
    v49 = *(_QWORD *)(a1 + 16);
    v50 = (v47 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v46 = *(_DWORD *)(a1 + 24) + 1;
    v44 = (__int64 *)(v49 + 16LL * v50);
    v51 = *v44;
    if ( v6 == *v44 )
      goto LABEL_50;
    v52 = 1;
    v53 = 0;
    while ( v51 != -8 )
    {
      if ( !v53 && v51 == -16 )
        v53 = v44;
      v50 = v48 & (v52 + v50);
      v44 = (__int64 *)(v49 + 16LL * v50);
      v51 = *v44;
      if ( v6 == *v44 )
        goto LABEL_50;
      ++v52;
    }
LABEL_58:
    if ( v53 )
      v44 = v53;
LABEL_50:
    *(_DWORD *)(a1 + 24) = v46;
    if ( *v44 != -8 )
      --*(_DWORD *)(a1 + 28);
    *v44 = v6;
    v14 = 0;
    *((_DWORD *)v44 + 2) = 0;
LABEL_10:
    if ( v7 == v14 )
    {
      v15 = *(_BYTE **)(a1 + 72);
      while ( 1 )
      {
        v20 = *(_QWORD *)(a1 + 48);
        v21 = (_QWORD *)(v20 - 8);
        if ( v15 == *(_BYTE **)(a1 + 80) )
        {
          sub_13725F0(v1, v15, v21);
          v22 = *(_BYTE **)(a1 + 72);
          v21 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
        }
        else
        {
          if ( v15 )
          {
            *(_QWORD *)v15 = *(_QWORD *)(v20 - 8);
            v15 = *(_BYTE **)(a1 + 72);
            v21 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          v22 = v15 + 8;
          *(_QWORD *)(a1 + 72) = v15 + 8;
        }
        v23 = *(_DWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 48) = v21;
        if ( !v23 )
          break;
        v16 = *((_QWORD *)v22 - 1);
        v17 = *(_QWORD *)(a1 + 16);
        v18 = (v23 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        result = (__int64 *)(v17 + 16LL * v18);
        v19 = *result;
        if ( *result != v16 )
        {
          v32 = 1;
          v33 = 0;
          while ( v19 != -8 )
          {
            if ( !v33 && v19 == -16 )
              v33 = result;
            v18 = (v23 - 1) & (v32 + v18);
            result = (__int64 *)(v17 + 16LL * v18);
            v19 = *result;
            if ( v16 == *result )
              goto LABEL_13;
            ++v32;
          }
          v34 = *(_DWORD *)(a1 + 24);
          if ( v33 )
            result = v33;
          ++*(_QWORD *)(a1 + 8);
          v28 = v34 + 1;
          if ( 4 * (v34 + 1) < 3 * v23 )
          {
            if ( v23 - *(_DWORD *)(a1 + 28) - v28 <= v23 >> 3 )
            {
              sub_13745E0(a1 + 8, v23);
              v35 = *(_DWORD *)(a1 + 32);
              if ( !v35 )
                goto LABEL_92;
              v36 = *((_QWORD *)v22 - 1);
              v37 = v35 - 1;
              v38 = *(_QWORD *)(a1 + 16);
              v39 = 0;
              v40 = 1;
              v28 = *(_DWORD *)(a1 + 24) + 1;
              v41 = v37 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
              result = (__int64 *)(v38 + 16LL * v41);
              v42 = *result;
              if ( *result != v36 )
              {
                while ( v42 != -8 )
                {
                  if ( v42 == -16 && !v39 )
                    v39 = result;
                  v41 = v37 & (v40 + v41);
                  result = (__int64 *)(v38 + 16LL * v41);
                  v42 = *result;
                  if ( v36 == *result )
                    goto LABEL_22;
                  ++v40;
                }
LABEL_36:
                if ( v39 )
                  result = v39;
              }
            }
LABEL_22:
            *(_DWORD *)(a1 + 24) = v28;
            if ( *result != -8 )
              --*(_DWORD *)(a1 + 28);
            v31 = *((_QWORD *)v22 - 1);
            *((_DWORD *)result + 2) = 0;
            *result = v31;
            goto LABEL_13;
          }
LABEL_20:
          sub_13745E0(a1 + 8, 2 * v23);
          v24 = *(_DWORD *)(a1 + 32);
          if ( !v24 )
            goto LABEL_92;
          v25 = *((_QWORD *)v22 - 1);
          v26 = v24 - 1;
          v27 = *(_QWORD *)(a1 + 16);
          v28 = *(_DWORD *)(a1 + 24) + 1;
          v29 = v26 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          result = (__int64 *)(v27 + 16LL * v29);
          v30 = *result;
          if ( *result != v25 )
          {
            v60 = 1;
            v39 = 0;
            while ( v30 != -8 )
            {
              if ( !v39 && v30 == -16 )
                v39 = result;
              v29 = v26 & (v60 + v29);
              result = (__int64 *)(v27 + 16LL * v29);
              v30 = *result;
              if ( v25 == *result )
                goto LABEL_22;
              ++v60;
            }
            goto LABEL_36;
          }
          goto LABEL_22;
        }
LABEL_13:
        *((_DWORD *)result + 2) = -1;
        v15 = *(_BYTE **)(a1 + 72);
        if ( v6 == *((_QWORD *)v15 - 1) )
          return result;
      }
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_20;
    }
  }
}
