// Function: sub_27F88F0
// Address: 0x27f88f0
//
__int64 __fastcall sub_27F88F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 i; // r13
  __int64 v9; // rbx
  __int64 v11; // rcx
  _QWORD *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // esi
  int v16; // r11d
  _QWORD *v17; // r10
  unsigned int v18; // edi
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 *v23; // rbx
  __int64 *v24; // r14
  __int64 *v25; // rdi
  __int64 v26; // rcx
  unsigned int v27; // esi
  int v28; // eax
  int v29; // r11d
  unsigned int v30; // edx
  __int64 *v31; // r10
  __int64 v32; // rsi
  int v33; // eax
  __int64 *v34; // rdi
  int v35; // r11d
  int v36; // eax
  int v37; // eax
  int v38; // r11d
  __int64 *v39; // rcx
  int v40; // esi
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r8d
  unsigned int v44; // edx
  __int64 v45; // rdi
  int v46; // esi
  _QWORD *v47; // rcx
  int v48; // edi
  int v49; // edi
  int v50; // edx
  _QWORD *v51; // rsi
  unsigned int v52; // r14d
  __int64 v53; // rcx
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
  if ( a3 == a2 )
    return result;
  i = a2;
  v54 = a1 + 32;
  v9 = *(_QWORD *)(a2 + 24);
LABEL_3:
  v11 = *(unsigned int *)(a1 + 40);
  result = *(_QWORD *)(a1 + 32);
  v12 = (_QWORD *)(result + 8 * v11);
  v13 = (8 * v11) >> 3;
  if ( !((8 * v11) >> 5) )
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
      if ( v9 == *(_QWORD *)result )
        goto LABEL_10;
      result += 8;
    }
    if ( v9 == *(_QWORD *)result )
      goto LABEL_10;
    result += 8;
LABEL_48:
    if ( v9 != *(_QWORD *)result )
    {
      v22 = v11 + 1;
      if ( v11 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
        goto LABEL_31;
      goto LABEL_50;
    }
    goto LABEL_10;
  }
  v14 = result + 32 * ((8 * v11) >> 5);
  while ( 1 )
  {
    if ( v9 == *(_QWORD *)result )
      goto LABEL_10;
    if ( v9 == *(_QWORD *)(result + 8) )
    {
      result += 8;
      goto LABEL_10;
    }
    if ( v9 == *(_QWORD *)(result + 16) )
    {
      result += 16;
      goto LABEL_10;
    }
    if ( v9 == *(_QWORD *)(result + 24) )
      break;
    result += 32;
    if ( result == v14 )
    {
      v13 = ((__int64)v12 - result) >> 3;
      goto LABEL_27;
    }
  }
  result += 24;
LABEL_10:
  if ( v12 != (_QWORD *)result )
    goto LABEL_11;
LABEL_30:
  v22 = v11 + 1;
  if ( v11 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
    goto LABEL_31;
LABEL_50:
  sub_C8D5F0(v54, v55, v22, 8u, a5, a6);
  v12 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
LABEL_31:
  *v12 = v9;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 8 )
  {
    v23 = *(__int64 **)(a1 + 32);
    v24 = &v23[result];
    while ( 1 )
    {
      v27 = *(_DWORD *)(a1 + 24);
      if ( !v27 )
        break;
      a6 = v27 - 1;
      a5 = *(_QWORD *)(a1 + 8);
      result = (unsigned int)a6 & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
      v25 = (__int64 *)(a5 + 8 * result);
      v26 = *v25;
      if ( *v23 != *v25 )
      {
        v35 = 1;
        v31 = 0;
        while ( v26 != -4096 )
        {
          if ( v31 || v26 != -8192 )
            v25 = v31;
          result = (unsigned int)a6 & (v35 + (_DWORD)result);
          v26 = *(_QWORD *)(a5 + 8LL * (unsigned int)result);
          if ( *v23 == v26 )
            goto LABEL_34;
          ++v35;
          v31 = v25;
          v25 = (__int64 *)(a5 + 8LL * (unsigned int)result);
        }
        v36 = *(_DWORD *)(a1 + 16);
        if ( !v31 )
          v31 = v25;
        ++*(_QWORD *)a1;
        v33 = v36 + 1;
        if ( 4 * v33 < 3 * v27 )
        {
          if ( v27 - *(_DWORD *)(a1 + 20) - v33 <= v27 >> 3 )
          {
            sub_27D4930(a1, v27);
            v37 = *(_DWORD *)(a1 + 24);
            if ( !v37 )
            {
LABEL_113:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            a5 = *v23;
            v38 = v37 - 1;
            a6 = *(_QWORD *)(a1 + 8);
            v39 = 0;
            v40 = 1;
            v41 = (v37 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
            v31 = (__int64 *)(a6 + 8LL * v41);
            v42 = *v31;
            v33 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v23 != *v31 )
            {
              while ( v42 != -4096 )
              {
                if ( !v39 && v42 == -8192 )
                  v39 = v31;
                v41 = v38 & (v40 + v41);
                v31 = (__int64 *)(a6 + 8LL * v41);
                v42 = *v31;
                if ( a5 == *v31 )
                  goto LABEL_60;
                ++v40;
              }
              if ( v39 )
                v31 = v39;
            }
          }
          goto LABEL_60;
        }
LABEL_37:
        sub_27D4930(a1, 2 * v27);
        v28 = *(_DWORD *)(a1 + 24);
        if ( !v28 )
          goto LABEL_113;
        v29 = v28 - 1;
        a6 = *(_QWORD *)(a1 + 8);
        v30 = (v28 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
        v31 = (__int64 *)(a6 + 8LL * v30);
        v32 = *v31;
        v33 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v23 != *v31 )
        {
          a5 = 1;
          v34 = 0;
          while ( v32 != -4096 )
          {
            if ( !v34 && v32 == -8192 )
              v34 = v31;
            v30 = v29 & (a5 + v30);
            v31 = (__int64 *)(a6 + 8LL * v30);
            v32 = *v31;
            if ( *v23 == *v31 )
              goto LABEL_60;
            a5 = (unsigned int)(a5 + 1);
          }
          if ( v34 )
            v31 = v34;
        }
LABEL_60:
        *(_DWORD *)(a1 + 16) = v33;
        if ( *v31 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = *v23;
        *v31 = *v23;
      }
LABEL_34:
      if ( v24 == ++v23 )
        goto LABEL_11;
    }
    ++*(_QWORD *)a1;
    goto LABEL_37;
  }
LABEL_11:
  for ( i = *(_QWORD *)(i + 8); i != a3; i = *(_QWORD *)(i + 8) )
  {
    result = *(unsigned int *)(a1 + 16);
    v9 = *(_QWORD *)(i + 24);
    if ( !(_DWORD)result )
      goto LABEL_3;
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      a6 = v15 - 1;
      a5 = *(_QWORD *)(a1 + 8);
      v16 = 1;
      v17 = 0;
      v18 = a6 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = (_QWORD *)(a5 + 8LL * v18);
      v20 = *v19;
      if ( v9 == *v19 )
        goto LABEL_11;
      while ( v20 != -4096 )
      {
        if ( v17 || v20 != -8192 )
          v19 = v17;
        v18 = a6 & (v16 + v18);
        v20 = *(_QWORD *)(a5 + 8LL * v18);
        if ( v9 == v20 )
          goto LABEL_11;
        ++v16;
        v17 = v19;
        v19 = (_QWORD *)(a5 + 8LL * v18);
      }
      if ( !v17 )
        v17 = v19;
      v21 = result + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v21 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 20) - v21 <= v15 >> 3 )
        {
          sub_27D4930(a1, v15);
          v48 = *(_DWORD *)(a1 + 24);
          if ( !v48 )
            goto LABEL_113;
          v49 = v48 - 1;
          a5 = *(_QWORD *)(a1 + 8);
          v50 = 1;
          v51 = 0;
          v52 = v49 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v17 = (_QWORD *)(a5 + 8LL * v52);
          v53 = *v17;
          v21 = *(_DWORD *)(a1 + 16) + 1;
          if ( v9 != *v17 )
          {
            while ( v53 != -4096 )
            {
              if ( v53 == -8192 && !v51 )
                v51 = v17;
              a6 = (unsigned int)(v50 + 1);
              v52 = v49 & (v50 + v52);
              v17 = (_QWORD *)(a5 + 8LL * v52);
              v53 = *v17;
              if ( v9 == *v17 )
                goto LABEL_20;
              ++v50;
            }
            if ( v51 )
              v17 = v51;
          }
        }
        goto LABEL_20;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_27D4930(a1, 2 * v15);
    v43 = *(_DWORD *)(a1 + 24);
    if ( !v43 )
      goto LABEL_113;
    a5 = (unsigned int)(v43 - 1);
    a6 = *(_QWORD *)(a1 + 8);
    v44 = a5 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v17 = (_QWORD *)(a6 + 8LL * v44);
    v45 = *v17;
    v21 = *(_DWORD *)(a1 + 16) + 1;
    if ( v9 != *v17 )
    {
      v46 = 1;
      v47 = 0;
      while ( v45 != -4096 )
      {
        if ( v45 == -8192 && !v47 )
          v47 = v17;
        v44 = a5 & (v46 + v44);
        v17 = (_QWORD *)(a6 + 8LL * v44);
        v45 = *v17;
        if ( v9 == *v17 )
          goto LABEL_20;
        ++v46;
      }
      if ( v47 )
        v17 = v47;
    }
LABEL_20:
    *(_DWORD *)(a1 + 16) = v21;
    if ( *v17 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *v17 = v9;
    result = *(unsigned int *)(a1 + 40);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(v54, v55, result + 1, 8u, a5, a6);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v9;
    ++*(_DWORD *)(a1 + 40);
  }
  return result;
}
