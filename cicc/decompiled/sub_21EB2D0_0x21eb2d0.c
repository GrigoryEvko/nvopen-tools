// Function: sub_21EB2D0
// Address: 0x21eb2d0
//
__int64 __fastcall sub_21EB2D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  int v7; // edx
  int v8; // edx
  unsigned int v9; // esi
  __int64 v10; // rax
  unsigned int v11; // ecx
  int v12; // edi
  char v13; // al
  int v14; // r8d
  unsigned int v15; // esi
  __int64 v16; // rdx
  unsigned int v17; // eax
  int v18; // ecx
  int v19; // r9d
  int *v20; // r9
  _DWORD *v21; // r9
  int v22; // eax
  int v23; // edi
  unsigned int v24; // r8d
  _DWORD *v25; // rcx
  int v26; // r9d
  int v27; // r9d
  int *v28; // r8
  _DWORD *v29; // r8
  int v30; // ecx
  int v31; // edi
  unsigned int v32; // r8d
  int *v33; // rdx
  int v34; // r9d
  int v35; // edx
  __int64 v36; // r11
  unsigned int v37; // r8d
  int v38; // edx
  int v39; // r10d
  _DWORD *v40; // r9
  _DWORD *v41; // r11
  int v42; // edx
  int v43; // edx
  __int64 v44; // r11
  _DWORD *v45; // r8
  int v46; // r9d
  unsigned int v47; // esi
  int v48; // r10d
  int *v49; // r11
  int v50; // eax
  int v51; // eax
  int v52; // [rsp+8h] [rbp-68h]
  int v53; // [rsp+8h] [rbp-68h]
  int v54; // [rsp+14h] [rbp-5Ch]
  int v55; // [rsp+14h] [rbp-5Ch]
  int v56; // [rsp+14h] [rbp-5Ch]
  int v57; // [rsp+14h] [rbp-5Ch]
  int v58; // [rsp+14h] [rbp-5Ch]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  int v62; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v63[7]; // [rsp+38h] [rbp-38h] BYREF

  v59 = a1 + 56;
  result = *(_QWORD *)a1 + 320LL;
  v60 = result;
  v61 = *(_QWORD *)(*(_QWORD *)a1 + 328LL);
  if ( v61 != result )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(v61 + 32);
      if ( v3 != v61 + 24 )
        break;
LABEL_23:
      result = *(_QWORD *)(v61 + 8);
      v61 = result;
      if ( v60 == result )
        return result;
    }
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 32);
      v5 = v4 + 40LL * *(unsigned int *)(v3 + 40);
      if ( v5 != v4 )
        break;
LABEL_21:
      if ( (*(_BYTE *)v3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v3 + 46) & 8) != 0 )
          v3 = *(_QWORD *)(v3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v61 + 24 == v3 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v4 )
        goto LABEL_10;
      v13 = *(_BYTE *)(v4 + 3);
      if ( (v13 & 0x20) != 0 )
        goto LABEL_10;
      v14 = *(_DWORD *)(v4 + 8);
      if ( v14 >= 0 )
        goto LABEL_10;
      if ( (v13 & 0x10) == 0 )
        goto LABEL_5;
      if ( **(_WORD **)(v3 + 16) != 45 && **(_WORD **)(v3 + 16) )
        goto LABEL_10;
      v15 = *(_DWORD *)(a1 + 80);
      v16 = *(_QWORD *)(a1 + 64);
      LODWORD(v63[0]) = *(_DWORD *)(v4 + 8);
      if ( !v15 )
        goto LABEL_30;
      v17 = (v15 - 1) & (37 * v14);
      v18 = *(_DWORD *)(v16 + 8LL * v17);
      if ( v14 != v18 )
        break;
LABEL_19:
      if ( (*(_BYTE *)(v4 + 3) & 0x10) != 0 )
        goto LABEL_10;
      v14 = *(_DWORD *)(v4 + 8);
LABEL_5:
      v6 = sub_1E69D00(*(_QWORD *)(a1 + 192), v14);
      if ( !v6 )
        goto LABEL_10;
      v7 = **(unsigned __int16 **)(v6 + 16);
      if ( (_WORD)v7 == 9 || *(_QWORD *)(v6 + 24) == v61 && **(_WORD **)(v6 + 16) && v7 != 45 )
        goto LABEL_10;
      v8 = *(_DWORD *)(v4 + 8);
      v9 = *(_DWORD *)(a1 + 80);
      v10 = *(_QWORD *)(a1 + 64);
      v62 = v8;
      if ( !v9 )
        goto LABEL_41;
      v11 = (v9 - 1) & (37 * v8);
      v12 = *(_DWORD *)(v10 + 8LL * v11);
      if ( v8 != v12 )
      {
        v27 = 1;
        while ( v12 != -1 )
        {
          v11 = (v9 - 1) & (v27 + v11);
          v12 = *(_DWORD *)(v10 + 8LL * v11);
          if ( v8 == v12 )
            goto LABEL_10;
          ++v27;
        }
LABEL_41:
        v28 = *(int **)(a1 + 96);
        if ( v28 == *(int **)(a1 + 104) )
        {
          sub_B8BBF0(a1 + 88, *(_BYTE **)(a1 + 96), &v62);
          v9 = *(_DWORD *)(a1 + 80);
          v10 = *(_QWORD *)(a1 + 64);
          v30 = ((__int64)(*(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88)) >> 2) - 1;
          if ( v9 )
            goto LABEL_45;
        }
        else
        {
          if ( v28 )
          {
            *v28 = v8;
            v28 = *(int **)(a1 + 96);
            v10 = *(_QWORD *)(a1 + 64);
            v9 = *(_DWORD *)(a1 + 80);
          }
          v29 = v28 + 1;
          *(_QWORD *)(a1 + 96) = v29;
          v30 = (((__int64)v29 - *(_QWORD *)(a1 + 88)) >> 2) - 1;
          if ( v9 )
          {
LABEL_45:
            v31 = v62;
            v32 = (v9 - 1) & (37 * v62);
            v33 = (int *)(v10 + 8LL * v32);
            v34 = *v33;
            if ( *v33 == v62 )
            {
LABEL_46:
              v33[1] = v30;
              goto LABEL_10;
            }
            v58 = 1;
            v49 = 0;
            while ( v34 != -1 )
            {
              if ( !v49 && v34 == -2 )
                v49 = v33;
              v32 = (v9 - 1) & (v58 + v32);
              v33 = (int *)(v10 + 8LL * v32);
              v34 = *v33;
              if ( v62 == *v33 )
                goto LABEL_46;
              ++v58;
            }
            v50 = *(_DWORD *)(a1 + 72);
            if ( v49 )
              v33 = v49;
            ++*(_QWORD *)(a1 + 56);
            v51 = v50 + 1;
            if ( 4 * v51 < 3 * v9 )
            {
              if ( v9 - (v51 + *(_DWORD *)(a1 + 76)) > v9 >> 3 )
              {
LABEL_80:
                *(_DWORD *)(a1 + 72) = v51;
                if ( *v33 != -1 )
                  --*(_DWORD *)(a1 + 76);
                *v33 = v31;
                v33[1] = 0;
                goto LABEL_46;
              }
              v55 = v30;
LABEL_86:
              sub_1BFDD60(v59, v9);
              sub_1BFD720(v59, &v62, v63);
              v33 = (int *)v63[0];
              v31 = v62;
              v30 = v55;
              v51 = *(_DWORD *)(a1 + 72) + 1;
              goto LABEL_80;
            }
LABEL_58:
            v55 = v30;
            v9 *= 2;
            goto LABEL_86;
          }
        }
        ++*(_QWORD *)(a1 + 56);
        goto LABEL_58;
      }
LABEL_10:
      v4 += 40;
      if ( v5 == v4 )
        goto LABEL_21;
    }
    v19 = 1;
    while ( v18 != -1 )
    {
      v17 = (v15 - 1) & (v19 + v17);
      v18 = *(_DWORD *)(v16 + 8LL * v17);
      if ( v14 == v18 )
        goto LABEL_19;
      ++v19;
    }
LABEL_30:
    v20 = *(int **)(a1 + 96);
    if ( v20 == *(int **)(a1 + 104) )
    {
      sub_B8BBF0(a1 + 88, *(_BYTE **)(a1 + 96), v63);
      v15 = *(_DWORD *)(a1 + 80);
      v16 = *(_QWORD *)(a1 + 64);
      v22 = ((__int64)(*(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88)) >> 2) - 1;
      if ( v15 )
        goto LABEL_34;
    }
    else
    {
      if ( v20 )
      {
        *v20 = v14;
        v20 = *(int **)(a1 + 96);
        v16 = *(_QWORD *)(a1 + 64);
        v15 = *(_DWORD *)(a1 + 80);
      }
      v21 = v20 + 1;
      *(_QWORD *)(a1 + 96) = v21;
      v22 = (((__int64)v21 - *(_QWORD *)(a1 + 88)) >> 2) - 1;
      if ( v15 )
      {
LABEL_34:
        v23 = v63[0];
        v24 = (v15 - 1) & (37 * LODWORD(v63[0]));
        v25 = (_DWORD *)(v16 + 8LL * v24);
        v26 = *v25;
        if ( *v25 == LODWORD(v63[0]) )
        {
LABEL_35:
          v25[1] = v22;
          goto LABEL_19;
        }
        v56 = 1;
        v41 = 0;
        while ( v26 != -1 )
        {
          if ( !v41 && v26 == -2 )
            v41 = v25;
          v24 = (v15 - 1) & (v56 + v24);
          v25 = (_DWORD *)(v16 + 8LL * v24);
          v26 = *v25;
          if ( LODWORD(v63[0]) == *v25 )
            goto LABEL_35;
          ++v56;
        }
        v42 = *(_DWORD *)(a1 + 72);
        if ( v41 )
          v25 = v41;
        ++*(_QWORD *)(a1 + 56);
        v38 = v42 + 1;
        if ( 4 * v38 < 3 * v15 )
        {
          if ( v15 - (v38 + *(_DWORD *)(a1 + 76)) <= v15 >> 3 )
          {
            v53 = v22;
            sub_1BFDD60(v59, v15);
            v43 = *(_DWORD *)(a1 + 80);
            if ( !v43 )
            {
LABEL_109:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
            v23 = v63[0];
            v44 = *(_QWORD *)(a1 + 64);
            v45 = 0;
            v57 = v43 - 1;
            v46 = 1;
            v47 = (v43 - 1) & (37 * LODWORD(v63[0]));
            v25 = (_DWORD *)(v44 + 8LL * v47);
            v38 = *(_DWORD *)(a1 + 72) + 1;
            v22 = v53;
            v48 = *v25;
            if ( *v25 != LODWORD(v63[0]) )
            {
              while ( v48 != -1 )
              {
                if ( !v45 && v48 == -2 )
                  v45 = v25;
                v47 = v57 & (v46 + v47);
                v25 = (_DWORD *)(v44 + 8LL * v47);
                v48 = *v25;
                if ( LODWORD(v63[0]) == *v25 )
                  goto LABEL_65;
                ++v46;
              }
              if ( v45 )
                v25 = v45;
            }
          }
          goto LABEL_65;
        }
LABEL_49:
        v52 = v22;
        sub_1BFDD60(v59, 2 * v15);
        v35 = *(_DWORD *)(a1 + 80);
        if ( !v35 )
          goto LABEL_109;
        v36 = *(_QWORD *)(a1 + 64);
        v54 = v35 - 1;
        v37 = (v35 - 1) & (37 * LODWORD(v63[0]));
        v25 = (_DWORD *)(v36 + 8LL * v37);
        v38 = *(_DWORD *)(a1 + 72) + 1;
        v22 = v52;
        v23 = *v25;
        if ( LODWORD(v63[0]) != *v25 )
        {
          v39 = 1;
          v40 = 0;
          while ( v23 != -1 )
          {
            if ( !v40 && v23 == -2 )
              v40 = v25;
            v37 = v54 & (v39 + v37);
            v25 = (_DWORD *)(v36 + 8LL * v37);
            v23 = *v25;
            if ( LODWORD(v63[0]) == *v25 )
              goto LABEL_65;
            ++v39;
          }
          v23 = v63[0];
          if ( v40 )
            v25 = v40;
        }
LABEL_65:
        *(_DWORD *)(a1 + 72) = v38;
        if ( *v25 != -1 )
          --*(_DWORD *)(a1 + 76);
        *v25 = v23;
        v25[1] = 0;
        goto LABEL_35;
      }
    }
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_49;
  }
  return result;
}
