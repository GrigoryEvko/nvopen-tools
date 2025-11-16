// Function: sub_20E8610
// Address: 0x20e8610
//
int *__fastcall sub_20E8610(__int64 a1, unsigned int a2)
{
  __int64 v2; // r14
  __int64 v4; // r9
  unsigned int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // r8
  int *result; // rax
  unsigned int v9; // edi
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // r10
  unsigned int m; // edx
  __int64 v15; // r10
  unsigned int v16; // edx
  unsigned int v17; // edx
  int v18; // r9d
  __int64 v19; // r14
  __int64 v20; // r11
  __int64 v21; // r10
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r15
  unsigned int i; // edi
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // ecx
  int v30; // ecx
  __int64 v31; // rdi
  int *v32; // r8
  int v33; // r15d
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  unsigned int k; // edx
  int v38; // esi
  unsigned int v39; // edx
  unsigned int v40; // edi
  int v41; // edx
  int v42; // ecx
  int v43; // edx
  int v44; // edx
  __int64 v45; // rsi
  int *v46; // rdi
  unsigned int v47; // r15d
  int j; // r8d
  int v49; // ecx
  unsigned int v50; // r15d
  int v51; // [rsp+Ch] [rbp-54h]
  int v52; // [rsp+Ch] [rbp-54h]
  __int64 v53; // [rsp+10h] [rbp-50h]
  int *v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+18h] [rbp-48h]
  __int64 v56; // [rsp+18h] [rbp-48h]
  int v57; // [rsp+20h] [rbp-40h]
  __int64 v58; // [rsp+20h] [rbp-40h]
  __int64 v59; // [rsp+20h] [rbp-40h]
  int v60; // [rsp+28h] [rbp-38h]
  unsigned __int64 v61; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 56);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(unsigned int *)(v4 + 4 * v2);
  result = *(int **)(a1 + 16);
  v9 = *(_DWORD *)(v4 + 4 * v2);
  v10 = 16 * v7;
  v11 = *(_QWORD *)&result[4 * v7];
  if ( !v5 )
  {
LABEL_7:
    v17 = *(_DWORD *)(v4 + 4LL * (unsigned int)(v2 + 1));
    if ( v9 >= v17 )
      return result;
    v53 = a1 + 32;
    v61 = (unsigned __int64)(unsigned int)(37 * v2) << 32;
    v18 = v2;
    v19 = *(_QWORD *)&result[(unsigned __int64)v10 / 4 + 2];
    v20 = 16 * (v7 + v17 - v9 - 1);
    v21 = 16 * (v9 + 1 - v7);
    if ( !v5 )
      goto LABEL_22;
LABEL_9:
    v57 = 1;
    v54 = 0;
    v22 = ((((unsigned int)(37 * v11) | v61) - 1 - ((unsigned __int64)(unsigned int)(37 * v11) << 32)) >> 22)
        ^ (((unsigned int)(37 * v11) | v61) - 1 - ((unsigned __int64)(unsigned int)(37 * v11) << 32));
    v23 = ((9 * (((v22 - 1 - (v22 << 13)) >> 8) ^ (v22 - 1 - (v22 << 13)))) >> 15)
        ^ (9 * (((v22 - 1 - (v22 << 13)) >> 8) ^ (v22 - 1 - (v22 << 13))));
    v24 = ((v23 - 1 - (v23 << 27)) >> 31) ^ (v23 - 1 - (v23 << 27));
    for ( i = v24 & (v5 - 1); ; i = (v5 - 1) & v40 )
    {
      result = (int *)(v6 + 24LL * i);
      v26 = *result;
      if ( v18 == *result && v11 == *((_QWORD *)result + 1) )
        goto LABEL_20;
      if ( v26 == -1 )
      {
        if ( *((_QWORD *)result + 1) == -1 )
        {
          if ( v54 )
            result = v54;
          v42 = *(_DWORD *)(a1 + 48);
          ++*(_QWORD *)(a1 + 32);
          v41 = v42 + 1;
          if ( 4 * (v42 + 1) < 3 * v5 )
          {
            if ( v5 - (v41 + *(_DWORD *)(a1 + 52)) > v5 >> 3 )
              goto LABEL_38;
            v52 = v18;
            v56 = v21;
            v59 = v20;
            sub_20E8370(v53, v5);
            v43 = *(_DWORD *)(a1 + 56);
            if ( !v43 )
            {
LABEL_64:
              ++*(_DWORD *)(a1 + 48);
              BUG();
            }
            v44 = v43 - 1;
            v45 = *(_QWORD *)(a1 + 40);
            v20 = v59;
            v46 = 0;
            v21 = v56;
            v18 = v52;
            v47 = v44 & v24;
            for ( j = 1; ; ++j )
            {
              result = (int *)(v45 + 24LL * v47);
              v49 = *result;
              if ( v52 == *result && v11 == *((_QWORD *)result + 1) )
                goto LABEL_44;
              if ( v49 == -1 )
              {
                if ( *((_QWORD *)result + 1) == -1 )
                {
                  if ( v46 )
                    result = v46;
                  v41 = *(_DWORD *)(a1 + 48) + 1;
                  goto LABEL_38;
                }
              }
              else if ( v49 == -2 && *((_QWORD *)result + 1) == -2 && !v46 )
              {
                v46 = (int *)(v45 + 24LL * v47);
              }
              v50 = j + v47;
              v47 = v44 & v50;
            }
          }
          while ( 1 )
          {
            v51 = v18;
            v55 = v21;
            v58 = v20;
            sub_20E8370(v53, 2 * v5);
            v29 = *(_DWORD *)(a1 + 56);
            if ( !v29 )
              goto LABEL_64;
            v30 = v29 - 1;
            v20 = v58;
            v21 = v55;
            v32 = 0;
            v18 = v51;
            v33 = 1;
            v34 = ((((unsigned int)(37 * v11) | v61) - 1 - ((unsigned __int64)(unsigned int)(37 * v11) << 32)) >> 22)
                ^ (((unsigned int)(37 * v11) | v61) - 1 - ((unsigned __int64)(unsigned int)(37 * v11) << 32));
            v35 = 9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13)));
            v36 = ((v35 >> 15) ^ v35) - 1 - (((v35 >> 15) ^ v35) << 27);
            for ( k = v30 & ((v36 >> 31) ^ v36); ; k = v30 & v39 )
            {
              v31 = *(_QWORD *)(a1 + 40);
              result = (int *)(v31 + 24LL * k);
              v38 = *result;
              if ( v51 == *result && v11 == *((_QWORD *)result + 1) )
                break;
              if ( v38 == -1 )
              {
                if ( *((_QWORD *)result + 1) == -1 )
                {
                  if ( v32 )
                    result = v32;
                  v41 = *(_DWORD *)(a1 + 48) + 1;
                  goto LABEL_38;
                }
              }
              else if ( v38 == -2 && *((_QWORD *)result + 1) == -2 && !v32 )
              {
                v32 = (int *)(v31 + 24LL * k);
              }
              v39 = v33 + k;
              ++v33;
            }
LABEL_44:
            v41 = *(_DWORD *)(a1 + 48) + 1;
LABEL_38:
            *(_DWORD *)(a1 + 48) = v41;
            if ( *result != -1 || *((_QWORD *)result + 1) != -1 )
              --*(_DWORD *)(a1 + 52);
            *result = v18;
            *((_QWORD *)result + 1) = v11;
            result[4] = 0;
LABEL_20:
            result[4] = v19;
            if ( v10 == v20 )
              return result;
            v27 = *(_QWORD *)(a1 + 16);
            v5 = *(_DWORD *)(a1 + 56);
            v6 = *(_QWORD *)(a1 + 40);
            v28 = v27 + v10;
            v10 += 16;
            v11 = *(_QWORD *)(v28 + v21);
            v19 = *(_QWORD *)(v27 + v10 + 8);
            if ( v5 )
              goto LABEL_9;
LABEL_22:
            ++*(_QWORD *)(a1 + 32);
          }
        }
      }
      else if ( v26 == -2 && *((_QWORD *)result + 1) == -2 )
      {
        if ( v54 )
          result = v54;
        v54 = result;
      }
      v40 = v57 + i;
      ++v57;
    }
  }
  v60 = 1;
  v12 = ((((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v2) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v11) << 32)) >> 22)
      ^ (((unsigned int)(37 * v11) | ((unsigned __int64)(unsigned int)(37 * v2) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v11) << 32));
  v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
      ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
  for ( m = (v5 - 1) & (((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - ((_DWORD)v13 << 27))); ; m = (v5 - 1) & v16 )
  {
    v15 = v6 + 24LL * m;
    if ( (_DWORD)v2 == *(_DWORD *)v15 && v11 == *(_QWORD *)(v15 + 8) )
      break;
    if ( *(_DWORD *)v15 == -1 && *(_QWORD *)(v15 + 8) == -1 )
      goto LABEL_7;
    v16 = v60 + m;
    ++v60;
  }
  return result;
}
