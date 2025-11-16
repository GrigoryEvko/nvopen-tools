// Function: sub_F774E0
// Address: 0xf774e0
//
__int64 __fastcall sub_F774E0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v8; // r12
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rcx
  __int64 *v17; // r11
  int v18; // r13d
  unsigned int v19; // edx
  __int64 *v20; // rdi
  int v21; // eax
  __int64 v22; // r13
  unsigned __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 *v25; // r14
  __int64 *v26; // rdi
  __int64 v27; // rcx
  unsigned int v28; // esi
  int v29; // eax
  int v30; // r11d
  unsigned int v31; // edx
  __int64 *v32; // r10
  __int64 v33; // rsi
  int v34; // eax
  __int64 *v35; // rdi
  int v36; // r11d
  int v37; // eax
  int v38; // eax
  int v39; // r11d
  __int64 *v40; // rcx
  int v41; // esi
  unsigned int v42; // edx
  __int64 v43; // rdi
  int v44; // r9d
  __int64 v45; // r10
  unsigned int v46; // edx
  int v47; // edi
  __int64 *v48; // rsi
  int v49; // r8d
  __int64 *v50; // r10
  int v51; // esi
  unsigned int v52; // edx
  __int64 v53; // rdi
  __int64 v54; // [rsp+10h] [rbp-40h]
  const void *v55; // [rsp+18h] [rbp-38h]

  v55 = (const void *)(a1 + 48);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  result = 0x800000000LL;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  if ( a2 == a3 )
    return result;
  v8 = a2;
  v54 = a1 + 32;
LABEL_3:
  v10 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v11 = *v8;
  v12 = (__int64 *)(result + 8 * v10);
  v13 = (8 * v10) >> 3;
  if ( !((8 * v10) >> 5) )
  {
LABEL_27:
    if ( v13 != 2 )
    {
      if ( v13 != 3 )
      {
        if ( v13 != 1 )
          goto LABEL_30;
        goto LABEL_48;
      }
      if ( v11 == *(_QWORD *)result )
        goto LABEL_10;
      result += 8;
    }
    if ( v11 == *(_QWORD *)result )
      goto LABEL_10;
    result += 8;
LABEL_48:
    if ( v11 != *(_QWORD *)result )
    {
      v23 = v10 + 1;
      if ( v10 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
        goto LABEL_31;
      goto LABEL_50;
    }
    goto LABEL_10;
  }
  v14 = result + 32 * ((8 * v10) >> 5);
  while ( 1 )
  {
    if ( v11 == *(_QWORD *)result )
      goto LABEL_10;
    if ( v11 == *(_QWORD *)(result + 8) )
    {
      result += 8;
      goto LABEL_10;
    }
    if ( v11 == *(_QWORD *)(result + 16) )
    {
      result += 16;
      goto LABEL_10;
    }
    if ( v11 == *(_QWORD *)(result + 24) )
      break;
    result += 32;
    if ( v14 == result )
    {
      v13 = ((__int64)v12 - result) >> 3;
      goto LABEL_27;
    }
  }
  result += 24;
LABEL_10:
  if ( v12 != (__int64 *)result )
    goto LABEL_11;
LABEL_30:
  v23 = v10 + 1;
  if ( v10 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
    goto LABEL_31;
LABEL_50:
  sub_C8D5F0(v54, v55, v23, 8u, a5, a6);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
LABEL_31:
  *v12 = v11;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 8 )
  {
    v24 = *(__int64 **)(a1 + 32);
    v25 = &v24[result];
    while ( 1 )
    {
      v28 = *(_DWORD *)(a1 + 24);
      if ( !v28 )
        break;
      a6 = v28 - 1;
      a5 = *(_QWORD *)(a1 + 8);
      result = (unsigned int)a6 & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
      v26 = (__int64 *)(a5 + 8 * result);
      v27 = *v26;
      if ( *v24 != *v26 )
      {
        v36 = 1;
        v32 = 0;
        while ( v27 != -4096 )
        {
          if ( v27 != -8192 || v32 )
            v26 = v32;
          result = (unsigned int)a6 & (v36 + (_DWORD)result);
          v27 = *(_QWORD *)(a5 + 8LL * (unsigned int)result);
          if ( *v24 == v27 )
            goto LABEL_34;
          ++v36;
          v32 = v26;
          v26 = (__int64 *)(a5 + 8LL * (unsigned int)result);
        }
        v37 = *(_DWORD *)(a1 + 16);
        if ( !v32 )
          v32 = v26;
        ++*(_QWORD *)a1;
        v34 = v37 + 1;
        if ( 4 * v34 < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 20) - v34 <= v28 >> 3 )
          {
            sub_CF28B0(a1, v28);
            v38 = *(_DWORD *)(a1 + 24);
            if ( !v38 )
            {
LABEL_113:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            a5 = *v24;
            v39 = v38 - 1;
            a6 = *(_QWORD *)(a1 + 8);
            v40 = 0;
            v41 = 1;
            v42 = (v38 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
            v32 = (__int64 *)(a6 + 8LL * v42);
            v43 = *v32;
            v34 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v24 != *v32 )
            {
              while ( v43 != -4096 )
              {
                if ( !v40 && v43 == -8192 )
                  v40 = v32;
                v42 = v39 & (v41 + v42);
                v32 = (__int64 *)(a6 + 8LL * v42);
                v43 = *v32;
                if ( a5 == *v32 )
                  goto LABEL_60;
                ++v41;
              }
              if ( v40 )
                v32 = v40;
            }
          }
          goto LABEL_60;
        }
LABEL_37:
        sub_CF28B0(a1, 2 * v28);
        v29 = *(_DWORD *)(a1 + 24);
        if ( !v29 )
          goto LABEL_113;
        v30 = v29 - 1;
        a6 = *(_QWORD *)(a1 + 8);
        v31 = (v29 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
        v32 = (__int64 *)(a6 + 8LL * v31);
        v33 = *v32;
        v34 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v24 != *v32 )
        {
          a5 = 1;
          v35 = 0;
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v35 )
              v35 = v32;
            v31 = v30 & (a5 + v31);
            v32 = (__int64 *)(a6 + 8LL * v31);
            v33 = *v32;
            if ( *v24 == *v32 )
              goto LABEL_60;
            a5 = (unsigned int)(a5 + 1);
          }
          if ( v35 )
            v32 = v35;
        }
LABEL_60:
        *(_DWORD *)(a1 + 16) = v34;
        if ( *v32 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = *v24;
        *v32 = *v24;
      }
LABEL_34:
      if ( v25 == ++v24 )
        goto LABEL_11;
    }
    ++*(_QWORD *)a1;
    goto LABEL_37;
  }
LABEL_11:
  for ( ++v8; a3 != v8; ++*(_DWORD *)(a1 + 40) )
  {
    result = *(unsigned int *)(a1 + 16);
    if ( !(_DWORD)result )
      goto LABEL_3;
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = *v8;
      a6 = *(_QWORD *)(a1 + 8);
      v17 = 0;
      v18 = 1;
      v19 = (v15 - 1) & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
      v20 = (__int64 *)(a6 + 8LL * v19);
      a5 = *v20;
      if ( *v8 == *v20 )
        goto LABEL_11;
      while ( a5 != -4096 )
      {
        if ( a5 != -8192 || v17 )
          v20 = v17;
        v19 = (v15 - 1) & (v18 + v19);
        a5 = *(_QWORD *)(a6 + 8LL * v19);
        if ( v16 == a5 )
          goto LABEL_11;
        ++v18;
        v17 = v20;
        v20 = (__int64 *)(a6 + 8LL * v19);
      }
      if ( !v17 )
        v17 = v20;
      v21 = result + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v21 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 20) - v21 <= v15 >> 3 )
        {
          sub_CF28B0(a1, v15);
          v49 = *(_DWORD *)(a1 + 24);
          if ( !v49 )
            goto LABEL_113;
          v16 = *v8;
          a5 = (unsigned int)(v49 - 1);
          a6 = *(_QWORD *)(a1 + 8);
          v50 = 0;
          v51 = 1;
          v52 = a5 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
          v17 = (__int64 *)(a6 + 8LL * v52);
          v53 = *v17;
          v21 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v17 != *v8 )
          {
            while ( v53 != -4096 )
            {
              if ( !v50 && v53 == -8192 )
                v50 = v17;
              v52 = a5 & (v51 + v52);
              v17 = (__int64 *)(a6 + 8LL * v52);
              v53 = *v17;
              if ( v16 == *v17 )
                goto LABEL_20;
              ++v51;
            }
            if ( v50 )
              v17 = v50;
          }
        }
        goto LABEL_20;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_CF28B0(a1, 2 * v15);
    v44 = *(_DWORD *)(a1 + 24);
    if ( !v44 )
      goto LABEL_113;
    a5 = *v8;
    a6 = (unsigned int)(v44 - 1);
    v45 = *(_QWORD *)(a1 + 8);
    v46 = a6 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
    v17 = (__int64 *)(v45 + 8LL * v46);
    v16 = *v17;
    v21 = *(_DWORD *)(a1 + 16) + 1;
    if ( *v8 != *v17 )
    {
      v47 = 1;
      v48 = 0;
      while ( v16 != -4096 )
      {
        if ( v16 == -8192 && !v48 )
          v48 = v17;
        v46 = a6 & (v47 + v46);
        v17 = (__int64 *)(v45 + 8LL * v46);
        v16 = *v17;
        if ( a5 == *v17 )
          goto LABEL_20;
        ++v47;
      }
      v16 = *v8;
      if ( v48 )
        v17 = v48;
    }
LABEL_20:
    *(_DWORD *)(a1 + 16) = v21;
    if ( *v17 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *v17 = v16;
    result = *(unsigned int *)(a1 + 40);
    v22 = *v8;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(v54, v55, result + 1, 8u, a5, a6);
      result = *(unsigned int *)(a1 + 40);
    }
    ++v8;
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v22;
  }
  return result;
}
