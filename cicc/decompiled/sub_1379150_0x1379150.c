// Function: sub_1379150
// Address: 0x1379150
//
__int64 __fastcall sub_1379150(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v8; // rdi
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r10d
  __int64 *v12; // r11
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned int j; // r8d
  __int64 *v17; // rcx
  __int64 v18; // r15
  unsigned int v19; // r8d
  unsigned int v20; // esi
  int v21; // edx
  __int64 result; // rax
  __int64 v23; // r12
  int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r8
  unsigned int v28; // edx
  __int64 v29; // rdi
  __int64 v30; // rcx
  int v31; // edx
  int v32; // r8d
  int v33; // ecx
  __int64 v34; // rdi
  __int64 v35; // r8
  int v36; // r9d
  unsigned int v37; // edx
  __int64 v38; // rsi
  int v39; // ecx
  int v40; // edi
  __int64 v41; // rdx
  int v42; // r8d
  __int64 *v43; // r9
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rsi
  unsigned int i; // eax
  __int64 v47; // rsi
  unsigned int v48; // eax
  int v49; // r10d
  int v50; // edi
  int v51; // edx
  int v52; // ecx
  __int64 v53; // rdi
  int v54; // r9d
  unsigned int v55; // edx
  __int64 v56; // rsi
  int v57; // edx
  int v58; // edx
  __int64 v59; // rdi
  int v60; // r8d
  unsigned int k; // eax
  __int64 v62; // rsi
  unsigned int v63; // eax
  int v64; // [rsp+8h] [rbp-68h]
  _QWORD v65[2]; // [rsp+18h] [rbp-58h] BYREF
  __int64 v66; // [rsp+28h] [rbp-48h]
  __int64 v67; // [rsp+30h] [rbp-40h]

  v8 = a1 + 32;
  v9 = *(_DWORD *)(a1 + 56);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 32);
LABEL_52:
    sub_1378AC0(v8, 2 * v9);
    v39 = *(_DWORD *)(a1 + 56);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 40);
      v42 = 1;
      v43 = 0;
      v44 = ((((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * a3) << 32)) >> 22)
          ^ (((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * a3) << 32));
      v45 = ((9 * (((v44 - 1 - (v44 << 13)) >> 8) ^ (v44 - 1 - (v44 << 13)))) >> 15)
          ^ (9 * (((v44 - 1 - (v44 << 13)) >> 8) ^ (v44 - 1 - (v44 << 13))));
      for ( i = (v39 - 1) & (((v45 - 1 - (v45 << 27)) >> 31) ^ (v45 - 1 - ((_DWORD)v45 << 27))); ; i = v40 & v48 )
      {
        v17 = (__int64 *)(v41 + 24LL * i);
        v47 = *v17;
        if ( a2 == *v17 && a3 == *((_DWORD *)v17 + 2) )
          break;
        if ( v47 == -8 )
        {
          if ( *((_DWORD *)v17 + 2) == -1 )
          {
LABEL_93:
            if ( v43 )
              v17 = v43;
            v32 = *(_DWORD *)(a1 + 48) + 1;
            goto LABEL_39;
          }
        }
        else if ( v47 == -16 && *((_DWORD *)v17 + 2) == -2 && !v43 )
        {
          v43 = (__int64 *)(v41 + 24LL * i);
        }
        v48 = v42 + i;
        ++v42;
      }
      goto LABEL_85;
    }
LABEL_104:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
  v10 = *(_QWORD *)(a1 + 40);
  v11 = 1;
  v12 = 0;
  v13 = ((((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * a3) << 32)) >> 22)
      ^ (((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * a3) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  v15 = ((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - (v14 << 27));
  for ( j = v15 & (v9 - 1); ; j = (v9 - 1) & v19 )
  {
    v17 = (__int64 *)(v10 + 24LL * j);
    v18 = *v17;
    if ( a2 == *v17 && a3 == *((_DWORD *)v17 + 2) )
      goto LABEL_11;
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && *((_DWORD *)v17 + 2) == -2 && !v12 )
      v12 = (__int64 *)(v10 + 24LL * j);
LABEL_9:
    v19 = v11 + j;
    ++v11;
  }
  if ( *((_DWORD *)v17 + 2) != -1 )
    goto LABEL_9;
  v31 = *(_DWORD *)(a1 + 48);
  if ( v12 )
    v17 = v12;
  ++*(_QWORD *)(a1 + 32);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v9 )
    goto LABEL_52;
  if ( v9 - *(_DWORD *)(a1 + 52) - v32 > v9 >> 3 )
    goto LABEL_39;
  v64 = v15;
  sub_1378AC0(v8, v9);
  v57 = *(_DWORD *)(a1 + 56);
  if ( !v57 )
    goto LABEL_104;
  v58 = v57 - 1;
  v43 = 0;
  v60 = 1;
  for ( k = v58 & v64; ; k = v58 & v63 )
  {
    v59 = *(_QWORD *)(a1 + 40);
    v17 = (__int64 *)(v59 + 24LL * k);
    v62 = *v17;
    if ( a2 == *v17 && a3 == *((_DWORD *)v17 + 2) )
      break;
    if ( v62 == -8 )
    {
      if ( *((_DWORD *)v17 + 2) == -1 )
        goto LABEL_93;
    }
    else if ( v62 == -16 && *((_DWORD *)v17 + 2) == -2 && !v43 )
    {
      v43 = (__int64 *)(v59 + 24LL * k);
    }
    v63 = v60 + k;
    ++v60;
  }
LABEL_85:
  v32 = *(_DWORD *)(a1 + 48) + 1;
LABEL_39:
  *(_DWORD *)(a1 + 48) = v32;
  if ( *v17 != -8 || *((_DWORD *)v17 + 2) != -1 )
    --*(_DWORD *)(a1 + 52);
  *v17 = a2;
  *((_DWORD *)v17 + 2) = a3;
  *((_DWORD *)v17 + 4) = -1;
LABEL_11:
  *((_DWORD *)v17 + 4) = a4;
  v65[0] = 2;
  v65[1] = 0;
  v66 = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220(v65);
  v20 = *(_DWORD *)(a1 + 24);
  v67 = a1;
  if ( !v20 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_16;
  }
  result = v66;
  v27 = *(_QWORD *)(a1 + 8);
  v28 = (v20 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
  v29 = v27 + 40LL * v28;
  v30 = *(_QWORD *)(v29 + 24);
  if ( v30 != v66 )
  {
    v49 = 1;
    v23 = 0;
    while ( v30 != -8 )
    {
      if ( v23 || v30 != -16 )
        v29 = v23;
      v28 = (v20 - 1) & (v49 + v28);
      v30 = *(_QWORD *)(v27 + 40LL * v28 + 24);
      if ( v66 == v30 )
        goto LABEL_30;
      ++v49;
      v23 = v29;
      v29 = v27 + 40LL * v28;
    }
    if ( !v23 )
      v23 = v29;
    v50 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v24 = v50 + 1;
    if ( 4 * (v50 + 1) >= 3 * v20 )
    {
LABEL_16:
      sub_1378D70(a1, 2 * v20);
      v21 = *(_DWORD *)(a1 + 24);
      if ( v21 )
      {
        result = v66;
        v33 = v21 - 1;
        v34 = *(_QWORD *)(a1 + 8);
        v35 = 0;
        v36 = 1;
        v37 = (v21 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v23 = v34 + 40LL * v37;
        v38 = *(_QWORD *)(v23 + 24);
        if ( v38 != v66 )
        {
          while ( v38 != -8 )
          {
            if ( v38 == -16 && !v35 )
              v35 = v23;
            v37 = v33 & (v36 + v37);
            v23 = v34 + 40LL * v37;
            v38 = *(_QWORD *)(v23 + 24);
            if ( v66 == v38 )
              goto LABEL_18;
            ++v36;
          }
LABEL_44:
          if ( v35 )
            v23 = v35;
        }
LABEL_18:
        v24 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_19;
      }
    }
    else
    {
      if ( v20 - *(_DWORD *)(a1 + 20) - v24 > v20 >> 3 )
      {
LABEL_19:
        *(_DWORD *)(a1 + 16) = v24;
        if ( *(_QWORD *)(v23 + 24) == -8 )
        {
          v26 = v23 + 8;
          if ( result != -8 )
          {
LABEL_24:
            *(_QWORD *)(v23 + 24) = result;
            if ( result != -8 && result != 0 && result != -16 )
              sub_1649AC0(v26, v65[0] & 0xFFFFFFFFFFFFFFF8LL);
            result = v66;
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v25 = *(_QWORD *)(v23 + 24);
          if ( result != v25 )
          {
            v26 = v23 + 8;
            if ( v25 != 0 && v25 != -8 && v25 != -16 )
            {
              sub_1649B30(v23 + 8);
              result = v66;
            }
            goto LABEL_24;
          }
        }
        *(_QWORD *)(v23 + 32) = v67;
        goto LABEL_30;
      }
      sub_1378D70(a1, v20);
      v51 = *(_DWORD *)(a1 + 24);
      if ( v51 )
      {
        result = v66;
        v52 = v51 - 1;
        v53 = *(_QWORD *)(a1 + 8);
        v35 = 0;
        v54 = 1;
        v55 = (v51 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v23 = v53 + 40LL * v55;
        v56 = *(_QWORD *)(v23 + 24);
        if ( v56 != v66 )
        {
          while ( v56 != -8 )
          {
            if ( !v35 && v56 == -16 )
              v35 = v23;
            v55 = v52 & (v54 + v55);
            v23 = v53 + 40LL * v55;
            v56 = *(_QWORD *)(v23 + 24);
            if ( v66 == v56 )
              goto LABEL_18;
            ++v54;
          }
          goto LABEL_44;
        }
        goto LABEL_18;
      }
    }
    result = v66;
    v23 = 0;
    goto LABEL_18;
  }
LABEL_30:
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v65);
  return result;
}
