// Function: sub_1D68B30
// Address: 0x1d68b30
//
__int64 __fastcall sub_1D68B30(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // r12
  _DWORD *v6; // rdi
  unsigned int i; // esi
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rax
  int v14; // ecx
  int v15; // r11d
  int *v16; // r10
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r8
  int j; // eax
  int *v21; // r8
  int v22; // r14d
  _DWORD *v24; // r14
  _DWORD *v25; // rax
  int v26; // ecx
  _DWORD *v27; // rbx
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rsi
  int v30; // r8d
  int k; // eax
  _DWORD *v32; // r12
  int v33; // eax
  __int64 v34; // r8
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rdi
  unsigned int v38; // eax
  unsigned __int64 v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rax
  _QWORD *v42; // rax
  int v43; // eax
  int v44; // eax
  int v45; // edx
  __int64 v46; // rax
  int v47; // eax
  int v48; // [rsp+10h] [rbp-80h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  unsigned __int64 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  int *v53; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v54; // [rsp+30h] [rbp-60h] BYREF
  __int64 v55; // [rsp+38h] [rbp-58h]
  __int64 v56; // [rsp+40h] [rbp-50h] BYREF
  _DWORD *v57; // [rsp+48h] [rbp-48h]
  __int64 v58; // [rsp+50h] [rbp-40h]
  unsigned int v59; // [rsp+58h] [rbp-38h]

  v2 = *a1;
  v3 = *((unsigned int *)a1 + 2);
  v56 = 0;
  v57 = 0;
  v4 = &v2[v3];
  v58 = 0;
  v59 = 0;
  if ( v2 == v4 )
  {
    v6 = 0;
    return j___libc_free_0(v6);
  }
  v6 = 0;
  for ( i = 0; ; i = v59 )
  {
    v8 = *v2;
    v9 = *(_DWORD *)(*v2 + 20) & 0xFFFFFFF;
    v10 = *(_QWORD *)(*v2 + 24 * (2 - v9));
    v11 = *(_QWORD **)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v11 = (_QWORD *)*v11;
    v12 = *(_QWORD *)(v8 + 24 * (1 - v9));
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    v54 = __PAIR64__((unsigned int)v11, (unsigned int)v13);
    v14 = (int)v13;
    v55 = v8;
    if ( i )
    {
      v15 = 1;
      v16 = 0;
      v17 = ((unsigned int)(37 * (_DWORD)v11) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v13) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * (_DWORD)v11) << 32);
      v18 = ((v17 >> 22) ^ v17) - 1 - (((v17 >> 22) ^ v17) << 13);
      v19 = ((9 * ((v18 >> 8) ^ v18)) >> 15) ^ (9 * ((v18 >> 8) ^ v18));
      for ( j = (i - 1) & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; j = (i - 1) & v47 )
      {
        v21 = &v6[4 * j];
        v22 = *v21;
        if ( v14 == *v21 && (_DWORD)v11 == v21[1] )
          break;
        if ( v22 == -1 )
        {
          if ( v21[1] == -1 )
          {
            if ( !v16 )
              v16 = &v6[4 * j];
            ++v56;
            v43 = v58 + 1;
            if ( 4 * ((int)v58 + 1) >= 3 * i )
              goto LABEL_51;
            if ( i - (v43 + HIDWORD(v58)) > i >> 3 )
              goto LABEL_53;
            goto LABEL_52;
          }
        }
        else if ( v22 == -2 && v21[1] == -2 && !v16 )
        {
          v16 = &v6[4 * j];
        }
        v47 = v15 + j;
        ++v15;
      }
    }
    else
    {
      ++v56;
LABEL_51:
      i *= 2;
LABEL_52:
      sub_1D685B0((__int64)&v56, i);
      sub_1D66860((__int64)&v56, (int *)&v54, &v53);
      v16 = v53;
      v14 = v54;
      v43 = v58 + 1;
LABEL_53:
      LODWORD(v58) = v43;
      if ( *v16 != -1 || v16[1] != -1 )
        --HIDWORD(v58);
      *v16 = v14;
      v16[1] = HIDWORD(v54);
      *((_QWORD *)v16 + 1) = v55;
      v6 = v57;
    }
    if ( v4 == ++v2 )
      break;
  }
  if ( !(_DWORD)v58 )
    return j___libc_free_0(v6);
  v24 = &v6[4 * v59];
  if ( v24 == v6 )
    return j___libc_free_0(v6);
  v25 = v6;
  while ( 1 )
  {
    v26 = *v25;
    v27 = v25;
    if ( *v25 != -1 )
      break;
    if ( v25[1] != -1 )
      goto LABEL_25;
LABEL_71:
    v25 += 4;
    if ( v24 == v25 )
      return j___libc_free_0(v6);
  }
  if ( v26 == -2 && v25[1] == -2 )
    goto LABEL_71;
LABEL_25:
  if ( v25 == v24 )
    return j___libc_free_0(v6);
  while ( 2 )
  {
    if ( v27[1] != v26 && v59 )
    {
      v28 = ((((unsigned int)(37 * v26) | ((unsigned __int64)(unsigned int)(37 * v26) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v26) << 32)) >> 22)
          ^ (((unsigned int)(37 * v26) | ((unsigned __int64)(unsigned int)(37 * v26) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v26) << 32));
      v29 = ((9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13)))) >> 15)
          ^ (9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13))));
      v30 = 1;
      for ( k = (v59 - 1) & (((v29 - 1 - (v29 << 27)) >> 31) ^ (v29 - 1 - ((_DWORD)v29 << 27))); ; k = (v59 - 1) & v33 )
      {
        v32 = &v6[4 * k];
        if ( *v32 == v26 && v32[1] == v26 )
          break;
        if ( *v32 == -1 && v32[1] == -1 )
          goto LABEL_39;
        v33 = v30 + k;
        ++v30;
      }
      v34 = *((_QWORD *)v27 + 1);
      if ( v32 != &v6[4 * v59] )
      {
        v35 = *(_DWORD *)(a2 + 24);
        if ( v35 )
        {
          v36 = *((_QWORD *)v32 + 1);
          v37 = *(_QWORD *)(a2 + 8);
          v38 = (v35 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v39 = v37 + 40LL * v38;
          v40 = *(_QWORD *)v39;
          if ( *(_QWORD *)v39 == v36 )
          {
LABEL_36:
            v41 = *(unsigned int *)(v39 + 16);
            if ( (unsigned int)v41 >= *(_DWORD *)(v39 + 20) )
            {
              v49 = *((_QWORD *)v27 + 1);
              sub_16CD150(v39 + 8, (const void *)(v39 + 24), 0, 8, v34, (_DWORD)v32 + 8);
              v34 = v49;
              v42 = (_QWORD *)(*(_QWORD *)(v39 + 8) + 8LL * *(unsigned int *)(v39 + 16));
            }
            else
            {
              v42 = (_QWORD *)(*(_QWORD *)(v39 + 8) + 8 * v41);
            }
            goto LABEL_38;
          }
          v48 = 1;
          v50 = 0;
          while ( v40 != -8 )
          {
            if ( v40 == -16 )
            {
              if ( v50 )
                v39 = v50;
              v50 = v39;
            }
            v38 = (v35 - 1) & (v48 + v38);
            v39 = v37 + 40LL * v38;
            v40 = *(_QWORD *)v39;
            if ( v36 == *(_QWORD *)v39 )
              goto LABEL_36;
            ++v48;
          }
          if ( v50 )
            v39 = v50;
          v44 = *(_DWORD *)(a2 + 16);
          ++*(_QWORD *)a2;
          v45 = v44 + 1;
          if ( 4 * (v44 + 1) < 3 * v35 )
          {
            if ( v35 - *(_DWORD *)(a2 + 20) - v45 <= v35 >> 3 )
            {
              v52 = v34;
              sub_1D68840(a2, v35);
              sub_1D67C70(a2, (__int64 *)v32 + 1, &v54);
              v39 = v54;
              v34 = v52;
              v45 = *(_DWORD *)(a2 + 16) + 1;
            }
            goto LABEL_65;
          }
        }
        else
        {
          ++*(_QWORD *)a2;
        }
        v51 = v34;
        sub_1D68840(a2, 2 * v35);
        sub_1D67C70(a2, (__int64 *)v32 + 1, &v54);
        v39 = v54;
        v34 = v51;
        v45 = *(_DWORD *)(a2 + 16) + 1;
LABEL_65:
        *(_DWORD *)(a2 + 16) = v45;
        if ( *(_QWORD *)v39 != -8 )
          --*(_DWORD *)(a2 + 20);
        v46 = *((_QWORD *)v32 + 1);
        *(_QWORD *)(v39 + 16) = 0x200000000LL;
        *(_QWORD *)v39 = v46;
        v42 = (_QWORD *)(v39 + 24);
        *(_QWORD *)(v39 + 8) = v39 + 24;
LABEL_38:
        *v42 = v34;
        v6 = v57;
        ++*(_DWORD *)(v39 + 16);
      }
    }
LABEL_39:
    v27 += 4;
    if ( v27 == v24 )
      return j___libc_free_0(v6);
    while ( 2 )
    {
      if ( *v27 == -1 )
      {
        if ( v27[1] != -1 )
          break;
        goto LABEL_45;
      }
      if ( *v27 == -2 && v27[1] == -2 )
      {
LABEL_45:
        v27 += 4;
        if ( v24 == v27 )
          return j___libc_free_0(v6);
        continue;
      }
      break;
    }
    if ( v27 != v24 )
    {
      v26 = *v27;
      continue;
    }
    return j___libc_free_0(v6);
  }
}
