// Function: sub_35D5240
// Address: 0x35d5240
//
__int64 __fastcall sub_35D5240(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r9d
  unsigned int i; // eax
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned __int16); // r15
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // rax
  __int64 v18; // rsi
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  unsigned int v23; // esi
  __int64 v24; // rdi
  int v25; // r11d
  __int64 *v26; // r10
  unsigned int j; // r8d
  __int64 *v28; // rcx
  __int64 v29; // r15
  unsigned int v30; // r8d
  int v31; // edi
  int v32; // r8d
  unsigned int v33; // esi
  __int64 v34; // rdi
  __int64 v35; // r8
  int v36; // r15d
  __int64 *v37; // r14
  unsigned int v38; // r10d
  __int64 *v39; // r9
  __int64 v40; // rcx
  unsigned int v41; // r10d
  int v42; // esi
  int v43; // esi
  __int64 v44; // rcx
  __int64 *v45; // r10
  int v46; // r8d
  unsigned int v47; // edx
  __int64 v48; // rdi
  unsigned int v49; // edx
  int v50; // ecx
  __int64 v51; // rsi
  int v52; // r10d
  __int64 *v53; // r9
  int v54; // r8d
  unsigned int v55; // edx
  __int64 v56; // rdi
  unsigned int v57; // edx
  int v58; // ecx
  int v59; // esi
  __int64 v60; // r9
  int v61; // r10d
  unsigned int k; // edx
  __int64 *v63; // rdi
  __int64 v64; // r8
  unsigned int v65; // edx
  int v66; // ecx
  int v67; // r8d
  int v68; // ecx
  int v69; // esi
  __int64 v70; // r8
  int v71; // r10d
  unsigned int m; // edx
  __int64 *v73; // rcx
  __int64 v74; // rdi
  unsigned int v75; // ecx
  unsigned int v76; // [rsp+Ch] [rbp-34h]
  unsigned int v77; // [rsp+Ch] [rbp-34h]
  unsigned int v78; // [rsp+Ch] [rbp-34h]
  unsigned int v79; // [rsp+Ch] [rbp-34h]

  v6 = *(unsigned int *)(a1 + 56);
  v7 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v6 )
  {
    v8 = 1;
    for ( i = (v6 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v6 - 1) & v11 )
    {
      v10 = v7 + 24LL * i;
      if ( a2 == *(_QWORD *)v10 && a3 == *(_QWORD *)(v10 + 8) )
        break;
      if ( *(_QWORD *)v10 == -4096 && *(_QWORD *)(v10 + 8) == -4096 )
        goto LABEL_10;
      v11 = v8 + i;
      ++v8;
    }
    if ( v10 != v7 + 24 * v6 )
      return *(unsigned int *)(v10 + 16);
  }
LABEL_10:
  v13 = sub_2E79000(*(__int64 **)a1);
  v14 = *(_QWORD *)(a1 + 16);
  v15 = v13;
  v16 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v14 + 552LL);
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v14 + 32LL);
  if ( v17 == sub_2D42F30 )
  {
    v18 = 2;
    v19 = sub_AE2980(v15, 0)[1];
    if ( v19 != 1 )
    {
      v18 = 3;
      if ( v19 != 2 )
      {
        v18 = 4;
        if ( v19 != 4 )
        {
          v18 = 5;
          if ( v19 != 8 )
          {
            v18 = 6;
            if ( v19 != 16 )
            {
              v18 = 7;
              if ( v19 != 32 )
              {
                v18 = 8;
                if ( v19 != 64 )
                  v18 = 9 * (unsigned int)(v19 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v18 = (unsigned int)v17(*(_QWORD *)(a1 + 16), v15, 0);
  }
  if ( v16 == sub_2EC09E0 )
    v22 = *(_QWORD *)(v14 + 8LL * (unsigned __int16)v18 + 112);
  else
    v22 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v16)(v14, v18, 0);
  result = sub_2EC06C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL), v22, byte_3F871B3, 0, v20, v21);
  v23 = *(_DWORD *)(a1 + 56);
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_66;
  }
  v24 = *(_QWORD *)(a1 + 40);
  v25 = 1;
  v26 = 0;
  for ( j = (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4))))
          & (v23 - 1); ; j = (v23 - 1) & v30 )
  {
    v28 = (__int64 *)(v24 + 24LL * j);
    v29 = *v28;
    if ( a2 == *v28 && a3 == v28[1] )
      goto LABEL_38;
    if ( v29 == -4096 )
      break;
    if ( v29 == -8192 && v28[1] == -8192 && !v26 )
      v26 = (__int64 *)(v24 + 24LL * j);
LABEL_29:
    v30 = v25 + j;
    ++v25;
  }
  if ( v28[1] != -4096 )
    goto LABEL_29;
  v31 = *(_DWORD *)(a1 + 48);
  if ( v26 )
    v28 = v26;
  ++*(_QWORD *)(a1 + 32);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) < 3 * v23 )
  {
    if ( v23 - *(_DWORD *)(a1 + 52) - v32 > v23 >> 3 )
      goto LABEL_35;
    v78 = result;
    sub_35D4A00(a1 + 32, v23);
    v58 = *(_DWORD *)(a1 + 56);
    if ( v58 )
    {
      v59 = v58 - 1;
      v60 = *(_QWORD *)(a1 + 40);
      v28 = 0;
      result = v78;
      v61 = 1;
      for ( k = v59
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; k = v59 & v65 )
      {
        v63 = (__int64 *)(v60 + 24LL * k);
        v64 = *v63;
        if ( a2 == *v63 && a3 == v63[1] )
        {
          v32 = *(_DWORD *)(a1 + 48) + 1;
          v28 = (__int64 *)(v60 + 24LL * k);
          goto LABEL_35;
        }
        if ( v64 == -4096 )
        {
          if ( v63[1] == -4096 )
          {
            if ( !v28 )
              v28 = (__int64 *)(v60 + 24LL * k);
            v32 = *(_DWORD *)(a1 + 48) + 1;
            goto LABEL_35;
          }
        }
        else if ( v64 == -8192 && v63[1] == -8192 && !v28 )
        {
          v28 = (__int64 *)(v60 + 24LL * k);
        }
        v65 = v61 + k;
        ++v61;
      }
    }
LABEL_128:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_66:
  v77 = result;
  sub_35D4A00(a1 + 32, 2 * v23);
  v50 = *(_DWORD *)(a1 + 56);
  if ( !v50 )
    goto LABEL_128;
  result = v77;
  v52 = 1;
  v53 = 0;
  v54 = v50 - 1;
  v55 = (v50 - 1)
      & (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
          | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4))));
  while ( 2 )
  {
    v51 = *(_QWORD *)(a1 + 40);
    v28 = (__int64 *)(v51 + 24LL * v55);
    v56 = *v28;
    if ( a2 == *v28 && a3 == v28[1] )
    {
      v32 = *(_DWORD *)(a1 + 48) + 1;
      goto LABEL_35;
    }
    if ( v56 != -4096 )
    {
      if ( v56 == -8192 && v28[1] == -8192 && !v53 )
        v53 = (__int64 *)(v51 + 24LL * v55);
      goto LABEL_74;
    }
    if ( v28[1] != -4096 )
    {
LABEL_74:
      v57 = v52 + v55;
      ++v52;
      v55 = v54 & v57;
      continue;
    }
    break;
  }
  if ( v53 )
    v28 = v53;
  v32 = *(_DWORD *)(a1 + 48) + 1;
LABEL_35:
  *(_DWORD *)(a1 + 48) = v32;
  if ( *v28 != -4096 || v28[1] != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v28 = a2;
  v28[1] = a3;
  *((_DWORD *)v28 + 4) = 0;
LABEL_38:
  *((_DWORD *)v28 + 4) = result;
  v33 = *(_DWORD *)(a1 + 88);
  v34 = a1 + 64;
  if ( !v33 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_56;
  }
  v35 = *(_QWORD *)(a1 + 72);
  v36 = 1;
  v37 = 0;
  v38 = (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
          | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4))))
      & (v33 - 1);
  while ( 2 )
  {
    v39 = (__int64 *)(v35 + 24LL * v38);
    v40 = *v39;
    if ( a2 == *v39 && a3 == v39[1] )
      goto LABEL_48;
    if ( v40 != -4096 )
    {
      if ( v40 == -8192 && v39[1] == -8192 && !v37 )
        v37 = (__int64 *)(v35 + 24LL * v38);
      goto LABEL_46;
    }
    if ( v39[1] != -4096 )
    {
LABEL_46:
      v41 = v36 + v38;
      ++v36;
      v38 = (v33 - 1) & v41;
      continue;
    }
    break;
  }
  v66 = *(_DWORD *)(a1 + 80);
  if ( v37 )
    v39 = v37;
  ++*(_QWORD *)(a1 + 64);
  v67 = v66 + 1;
  if ( 4 * (v66 + 1) < 3 * v33 )
  {
    if ( v33 - *(_DWORD *)(a1 + 84) - v67 > v33 >> 3 )
      goto LABEL_89;
    v79 = result;
    sub_35D4A00(v34, v33);
    v68 = *(_DWORD *)(a1 + 88);
    if ( v68 )
    {
      v69 = v68 - 1;
      v39 = 0;
      result = v79;
      v71 = 1;
      for ( m = (v68 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = v69 & v75 )
      {
        v70 = *(_QWORD *)(a1 + 72);
        v73 = (__int64 *)(v70 + 24LL * m);
        v74 = *v73;
        if ( a2 == *v73 && a3 == v73[1] )
        {
          v39 = (__int64 *)(v70 + 24LL * m);
          v67 = *(_DWORD *)(a1 + 80) + 1;
          goto LABEL_89;
        }
        if ( v74 == -4096 )
        {
          if ( v73[1] == -4096 )
          {
            if ( !v39 )
              v39 = (__int64 *)(v70 + 24LL * m);
            v67 = *(_DWORD *)(a1 + 80) + 1;
            goto LABEL_89;
          }
        }
        else if ( v74 == -8192 && v73[1] == -8192 && !v39 )
        {
          v39 = (__int64 *)(v70 + 24LL * m);
        }
        v75 = m + v71++;
      }
    }
LABEL_127:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
LABEL_56:
  v76 = result;
  sub_35D4A00(v34, 2 * v33);
  v42 = *(_DWORD *)(a1 + 88);
  if ( !v42 )
    goto LABEL_127;
  v43 = v42 - 1;
  result = v76;
  v45 = 0;
  v46 = 1;
  v47 = v43
      & (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
          | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4))));
  while ( 2 )
  {
    v44 = *(_QWORD *)(a1 + 72);
    v39 = (__int64 *)(v44 + 24LL * v47);
    v48 = *v39;
    if ( a2 == *v39 && a3 == v39[1] )
    {
      v67 = *(_DWORD *)(a1 + 80) + 1;
      goto LABEL_89;
    }
    if ( v48 != -4096 )
    {
      if ( v48 == -8192 && v39[1] == -8192 && !v45 )
        v45 = (__int64 *)(v44 + 24LL * v47);
      goto LABEL_64;
    }
    if ( v39[1] != -4096 )
    {
LABEL_64:
      v49 = v46 + v47;
      ++v46;
      v47 = v43 & v49;
      continue;
    }
    break;
  }
  if ( v45 )
    v39 = v45;
  v67 = *(_DWORD *)(a1 + 80) + 1;
LABEL_89:
  *(_DWORD *)(a1 + 80) = v67;
  if ( *v39 != -4096 || v39[1] != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v39 = a2;
  v39[1] = a3;
  *((_DWORD *)v39 + 4) = 0;
LABEL_48:
  *((_DWORD *)v39 + 4) = result;
  return result;
}
