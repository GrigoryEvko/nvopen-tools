// Function: sub_FF5F90
// Address: 0xff5f90
//
__int64 __fastcall sub_FF5F90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  unsigned int v6; // esi
  int v7; // edx
  __int64 v8; // rax
  _QWORD *v9; // rbx
  int v10; // ecx
  __int64 v11; // rdx
  unsigned __int64 *v12; // r12
  __int64 v13; // r8
  unsigned int v14; // edx
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 result; // rax
  unsigned int v18; // r14d
  int v19; // r12d
  unsigned int v20; // esi
  _DWORD *v21; // rdx
  __int64 v22; // r9
  unsigned __int64 v23; // rbx
  __int64 *v24; // rax
  unsigned int j; // r8d
  __int64 *v26; // rcx
  __int64 v27; // rdi
  _DWORD *v28; // rax
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  int v32; // r10d
  __int64 *v33; // r9
  unsigned int i; // ecx
  __int64 v35; // rdi
  unsigned int v36; // ecx
  unsigned int v37; // r8d
  int v38; // edi
  int v39; // ecx
  int v40; // ecx
  int v41; // ecx
  __int64 v42; // rdi
  __int64 *v43; // r8
  unsigned int v44; // ebx
  int k; // r9d
  __int64 v46; // rsi
  unsigned int v47; // ebx
  int v48; // ecx
  __int64 v49; // r8
  _QWORD *v50; // r9
  unsigned int v51; // edx
  int v52; // esi
  __int64 v53; // rdi
  int v54; // r10d
  int v55; // edi
  int v56; // edx
  int v57; // ecx
  __int64 v58; // r8
  unsigned int v59; // edx
  int v60; // esi
  __int64 v61; // rdi
  __int64 v62; // [rsp+0h] [rbp-80h]
  __int64 v63; // [rsp+0h] [rbp-80h]
  __int64 v64; // [rsp+8h] [rbp-78h]
  int v65; // [rsp+10h] [rbp-70h]
  _DWORD *v66; // [rsp+10h] [rbp-70h]
  _DWORD *v67; // [rsp+10h] [rbp-70h]
  __int64 v68; // [rsp+18h] [rbp-68h]
  __int64 v69; // [rsp+18h] [rbp-68h]
  __int64 v70; // [rsp+18h] [rbp-68h]
  __int64 v71; // [rsp+18h] [rbp-68h]
  unsigned __int64 v72; // [rsp+18h] [rbp-68h]
  __int64 v73; // [rsp+18h] [rbp-68h]
  _QWORD v74[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v75; // [rsp+38h] [rbp-48h]
  __int64 v76; // [rsp+40h] [rbp-40h]

  v3 = a2;
  v74[0] = 2;
  v74[1] = 0;
  v75 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
  {
    sub_BD73F0((__int64)v74);
    v3 = a2;
  }
  v6 = *(_DWORD *)(a1 + 24);
  v76 = a1;
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_6;
  }
  v8 = v75;
  v13 = *(_QWORD *)(a1 + 8);
  v14 = (v6 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
  v15 = (_QWORD *)(v13 + 40LL * v14);
  v16 = v15[3];
  if ( v75 != v16 )
  {
    v54 = 1;
    v9 = 0;
    while ( v16 != -4096 )
    {
      if ( v9 || v16 != -8192 )
        v15 = v9;
      v14 = (v6 - 1) & (v54 + v14);
      v16 = *(_QWORD *)(v13 + 40LL * v14 + 24);
      if ( v75 == v16 )
        goto LABEL_19;
      ++v54;
      v9 = v15;
      v15 = (_QWORD *)(v13 + 40LL * v14);
    }
    if ( !v9 )
      v9 = v15;
    v55 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v10 = v55 + 1;
    if ( 4 * (v55 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 20) - v10 > v6 >> 3 )
      {
LABEL_9:
        *(_DWORD *)(a1 + 16) = v10;
        if ( v9[3] == -4096 )
        {
          v12 = v9 + 1;
          if ( v8 != -4096 )
          {
LABEL_14:
            v9[3] = v8;
            if ( v8 == -4096 || v8 == 0 || v8 == -8192 )
            {
              v8 = v75;
            }
            else
            {
              v70 = v3;
              sub_BD6050(v12, v74[0] & 0xFFFFFFFFFFFFFFF8LL);
              v8 = v75;
              v3 = v70;
            }
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v11 = v9[3];
          if ( v8 != v11 )
          {
            v12 = v9 + 1;
            if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
            {
              v69 = v3;
              sub_BD60C0(v9 + 1);
              v8 = v75;
              v3 = v69;
            }
            goto LABEL_14;
          }
        }
        v9[4] = v76;
        goto LABEL_19;
      }
      v73 = v3;
      sub_FF5150(a1, v6);
      v56 = *(_DWORD *)(a1 + 24);
      v3 = v73;
      if ( !v56 )
        goto LABEL_7;
      v8 = v75;
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a1 + 8);
      v50 = 0;
      v59 = (v56 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
      v9 = (_QWORD *)(v58 + 40LL * v59);
      v60 = 1;
      v61 = v9[3];
      if ( v75 == v61 )
        goto LABEL_8;
      while ( v61 != -4096 )
      {
        if ( !v50 && v61 == -8192 )
          v50 = v9;
        v59 = v57 & (v60 + v59);
        v9 = (_QWORD *)(v58 + 40LL * v59);
        v61 = v9[3];
        if ( v75 == v61 )
          goto LABEL_8;
        ++v60;
      }
      goto LABEL_72;
    }
LABEL_6:
    v68 = v3;
    sub_FF5150(a1, 2 * v6);
    v7 = *(_DWORD *)(a1 + 24);
    v3 = v68;
    if ( !v7 )
    {
LABEL_7:
      v8 = v75;
      v9 = 0;
LABEL_8:
      v10 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_9;
    }
    v8 = v75;
    v48 = v7 - 1;
    v49 = *(_QWORD *)(a1 + 8);
    v50 = 0;
    v51 = (v7 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
    v9 = (_QWORD *)(v49 + 40LL * v51);
    v52 = 1;
    v53 = v9[3];
    if ( v53 == v75 )
      goto LABEL_8;
    while ( v53 != -4096 )
    {
      if ( v53 == -8192 && !v50 )
        v50 = v9;
      v51 = v48 & (v52 + v51);
      v9 = (_QWORD *)(v49 + 40LL * v51);
      v53 = v9[3];
      if ( v75 == v53 )
        goto LABEL_8;
      ++v52;
    }
LABEL_72:
    if ( v50 )
      v9 = v50;
    goto LABEL_8;
  }
LABEL_19:
  if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
  {
    v71 = v3;
    sub_BD60C0(v74);
    v3 = v71;
  }
  result = *(unsigned int *)(a3 + 8);
  if ( !(_DWORD)result )
    return result;
  v18 = 0;
  v19 = 0;
  v72 = (unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32;
  v64 = a1 + 32;
  result = 0;
  do
  {
    v20 = *(_DWORD *)(a1 + 56);
    v21 = (_DWORD *)(*(_QWORD *)a3 + 4 * result);
    if ( !v20 )
    {
      ++*(_QWORD *)(a1 + 32);
LABEL_38:
      v62 = v3;
      v66 = v21;
      sub_FF1CF0(v64, 2 * v20);
      v29 = *(_DWORD *)(a1 + 56);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 40);
        v21 = v66;
        v3 = v62;
        v32 = 1;
        v33 = 0;
        for ( i = v30 & (((0xBF58476D1CE4E5B9LL * (v72 | v18)) >> 31) ^ (484763065 * (v72 | v18))); ; i = v30 & v36 )
        {
          v24 = (__int64 *)(v31 + 24LL * i);
          v35 = *v24;
          if ( v62 == *v24 && v19 == *((_DWORD *)v24 + 2) )
            break;
          if ( v35 == -4096 )
          {
            if ( *((_DWORD *)v24 + 2) == -1 )
            {
              if ( v33 )
                v24 = v33;
              v39 = *(_DWORD *)(a1 + 48) + 1;
              goto LABEL_54;
            }
          }
          else if ( v35 == -8192 && *((_DWORD *)v24 + 2) == -2 && !v33 )
          {
            v33 = (__int64 *)(v31 + 24LL * i);
          }
          v36 = v32 + i;
          ++v32;
        }
        goto LABEL_69;
      }
LABEL_110:
      ++*(_DWORD *)(a1 + 48);
      BUG();
    }
    v65 = 1;
    v22 = *(_QWORD *)(a1 + 40);
    v23 = ((0xBF58476D1CE4E5B9LL * (v72 | v18)) >> 31) ^ (0xBF58476D1CE4E5B9LL * (v72 | v18));
    v24 = 0;
    for ( j = v23 & (v20 - 1); ; j = (v20 - 1) & v37 )
    {
      v26 = (__int64 *)(v22 + 24LL * j);
      v27 = *v26;
      if ( v3 == *v26 && v19 == *((_DWORD *)v26 + 2) )
      {
        v28 = v26 + 2;
        goto LABEL_35;
      }
      if ( v27 == -4096 )
        break;
      if ( v27 == -8192 && *((_DWORD *)v26 + 2) == -2 && !v24 )
        v24 = (__int64 *)(v22 + 24LL * j);
LABEL_49:
      v37 = v65 + j;
      ++v65;
    }
    if ( *((_DWORD *)v26 + 2) != -1 )
      goto LABEL_49;
    v38 = *(_DWORD *)(a1 + 48);
    if ( !v24 )
      v24 = (__int64 *)(v22 + 24LL * j);
    ++*(_QWORD *)(a1 + 32);
    v39 = v38 + 1;
    if ( 4 * (v38 + 1) >= 3 * v20 )
      goto LABEL_38;
    if ( v20 - *(_DWORD *)(a1 + 52) - v39 <= v20 >> 3 )
    {
      v63 = v3;
      v67 = v21;
      sub_FF1CF0(v64, v20);
      v40 = *(_DWORD *)(a1 + 56);
      if ( v40 )
      {
        v41 = v40 - 1;
        v21 = v67;
        v43 = 0;
        v3 = v63;
        v44 = v41 & v23;
        for ( k = 1; ; ++k )
        {
          v42 = *(_QWORD *)(a1 + 40);
          v24 = (__int64 *)(v42 + 24LL * v44);
          v46 = *v24;
          if ( v63 == *v24 && v19 == *((_DWORD *)v24 + 2) )
            break;
          if ( v46 == -4096 )
          {
            if ( *((_DWORD *)v24 + 2) == -1 )
            {
              if ( v43 )
                v24 = v43;
              v39 = *(_DWORD *)(a1 + 48) + 1;
              goto LABEL_54;
            }
          }
          else if ( v46 == -8192 && *((_DWORD *)v24 + 2) == -2 && !v43 )
          {
            v43 = (__int64 *)(v42 + 24LL * v44);
          }
          v47 = k + v44;
          v44 = v41 & v47;
        }
LABEL_69:
        v39 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_54;
      }
      goto LABEL_110;
    }
LABEL_54:
    *(_DWORD *)(a1 + 48) = v39;
    if ( *v24 != -4096 || *((_DWORD *)v24 + 2) != -1 )
      --*(_DWORD *)(a1 + 52);
    *v24 = v3;
    v28 = v24 + 2;
    *(v28 - 2) = v19;
    *v28 = -1;
LABEL_35:
    v18 += 37;
    *v28 = *v21;
    result = (unsigned int)(v19 + 1);
    v19 = result;
  }
  while ( *(_DWORD *)(a3 + 8) > (unsigned int)result );
  return result;
}
