// Function: sub_351E510
// Address: 0x351e510
//
__int64 __fastcall sub_351E510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 i; // r12
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // r13
  _QWORD *v13; // rsi
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 *v20; // r11
  int v21; // r13d
  unsigned int v22; // edx
  __int64 *v23; // rdi
  __int64 v24; // r8
  int v25; // eax
  __int64 v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // r13
  _QWORD *v31; // r14
  __int64 v32; // r8
  unsigned int v33; // eax
  _QWORD *v34; // rdi
  __int64 v35; // rcx
  unsigned int v36; // esi
  int v37; // eax
  int v38; // r11d
  __int64 v39; // r9
  unsigned int v40; // edx
  _QWORD *v41; // r10
  __int64 v42; // rsi
  int v43; // eax
  int v44; // r8d
  _QWORD *v45; // rdi
  int v46; // r11d
  int v47; // eax
  int v48; // eax
  int v49; // r11d
  __int64 v50; // r9
  _QWORD *v51; // rcx
  int v52; // esi
  unsigned int v53; // edx
  __int64 v54; // rdi
  int v55; // r8d
  __int64 v56; // r10
  unsigned int v57; // edx
  __int64 v58; // rdi
  int v59; // esi
  __int64 *v60; // rcx
  int v61; // edi
  int v62; // edi
  __int64 *v63; // r10
  int v64; // ecx
  unsigned int v65; // edx
  __int64 v66; // rsi
  __int64 v67; // [rsp+10h] [rbp-40h]
  const void *v68; // [rsp+18h] [rbp-38h]

  v68 = (const void *)(a1 + 48);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  result = 0x1000000000LL;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0x1000000000LL;
  if ( a2 == a3 )
    return result;
  i = a2;
  v67 = a1 + 32;
LABEL_3:
  v10 = *(unsigned int *)(a1 + 40);
  v11 = *(_QWORD **)(a1 + 32);
  v12 = *(_QWORD *)(i + 32);
  v13 = &v11[v10];
  v14 = (8 * v10) >> 3;
  if ( !((8 * v10) >> 5) )
  {
LABEL_27:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_30;
        goto LABEL_48;
      }
      if ( *v11 == v12 )
        goto LABEL_10;
      ++v11;
    }
    if ( *v11 == v12 )
      goto LABEL_10;
    ++v11;
LABEL_48:
    if ( *v11 != v12 )
    {
      v28 = v10 + 1;
      if ( v10 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
        goto LABEL_31;
      goto LABEL_50;
    }
    goto LABEL_10;
  }
  v15 = &v11[4 * ((8 * v10) >> 5)];
  while ( 1 )
  {
    if ( *v11 == v12 )
      goto LABEL_10;
    if ( v11[1] == v12 )
    {
      ++v11;
      goto LABEL_10;
    }
    if ( v11[2] == v12 )
    {
      v11 += 2;
      goto LABEL_10;
    }
    if ( v11[3] == v12 )
      break;
    v11 += 4;
    if ( v15 == v11 )
    {
      v14 = v13 - v11;
      goto LABEL_27;
    }
  }
  v11 += 3;
LABEL_10:
  if ( v13 != v11 )
    goto LABEL_11;
LABEL_30:
  v28 = v10 + 1;
  if ( v10 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
    goto LABEL_31;
LABEL_50:
  sub_C8D5F0(v67, v68, v28, 8u, a5, a6);
  v13 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
LABEL_31:
  *v13 = v12;
  v29 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = v29;
  if ( (unsigned int)v29 > 0x10 )
  {
    v30 = *(_QWORD **)(a1 + 32);
    v31 = &v30[v29];
    while ( 1 )
    {
      v36 = *(_DWORD *)(a1 + 24);
      if ( !v36 )
        break;
      v32 = *(_QWORD *)(a1 + 8);
      v33 = (v36 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
      v34 = (_QWORD *)(v32 + 8LL * v33);
      v35 = *v34;
      if ( *v30 != *v34 )
      {
        v46 = 1;
        v41 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 != -8192 || v41 )
            v34 = v41;
          v33 = (v36 - 1) & (v46 + v33);
          v35 = *(_QWORD *)(v32 + 8LL * v33);
          if ( *v30 == v35 )
            goto LABEL_34;
          ++v46;
          v41 = v34;
          v34 = (_QWORD *)(v32 + 8LL * v33);
        }
        v47 = *(_DWORD *)(a1 + 16);
        if ( !v41 )
          v41 = v34;
        ++*(_QWORD *)a1;
        v43 = v47 + 1;
        if ( 4 * v43 < 3 * v36 )
        {
          if ( v36 - *(_DWORD *)(a1 + 20) - v43 <= v36 >> 3 )
          {
            sub_2E61F50(a1, v36);
            v48 = *(_DWORD *)(a1 + 24);
            if ( !v48 )
            {
LABEL_113:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v49 = v48 - 1;
            v50 = *(_QWORD *)(a1 + 8);
            v51 = 0;
            v52 = 1;
            v53 = (v48 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
            v41 = (_QWORD *)(v50 + 8LL * v53);
            v54 = *v41;
            v43 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v30 != *v41 )
            {
              while ( v54 != -4096 )
              {
                if ( !v51 && v54 == -8192 )
                  v51 = v41;
                v53 = v49 & (v52 + v53);
                v41 = (_QWORD *)(v50 + 8LL * v53);
                v54 = *v41;
                if ( *v30 == *v41 )
                  goto LABEL_60;
                ++v52;
              }
              if ( v51 )
                v41 = v51;
            }
          }
          goto LABEL_60;
        }
LABEL_37:
        sub_2E61F50(a1, 2 * v36);
        v37 = *(_DWORD *)(a1 + 24);
        if ( !v37 )
          goto LABEL_113;
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 8);
        v40 = (v37 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
        v41 = (_QWORD *)(v39 + 8LL * v40);
        v42 = *v41;
        v43 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v30 != *v41 )
        {
          v44 = 1;
          v45 = 0;
          while ( v42 != -4096 )
          {
            if ( v42 == -8192 && !v45 )
              v45 = v41;
            v40 = v38 & (v44 + v40);
            v41 = (_QWORD *)(v39 + 8LL * v40);
            v42 = *v41;
            if ( *v30 == *v41 )
              goto LABEL_60;
            ++v44;
          }
          if ( v45 )
            v41 = v45;
        }
LABEL_60:
        *(_DWORD *)(a1 + 16) = v43;
        if ( *v41 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v41 = *v30;
      }
LABEL_34:
      if ( v31 == ++v30 )
        goto LABEL_11;
    }
    ++*(_QWORD *)a1;
    goto LABEL_37;
  }
LABEL_11:
  result = sub_220EF30(i);
  for ( i = result; result != a3; i = result )
  {
    v16 = *(_DWORD *)(a1 + 16);
    if ( !v16 )
      goto LABEL_3;
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = *(_QWORD *)(i + 32);
      v19 = *(_QWORD *)(a1 + 8);
      v20 = 0;
      v21 = 1;
      v22 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v23 = (__int64 *)(v19 + 8LL * v22);
      v24 = *v23;
      if ( v18 == *v23 )
        goto LABEL_11;
      while ( v24 != -4096 )
      {
        if ( v24 != -8192 || v20 )
          v23 = v20;
        v22 = (v17 - 1) & (v21 + v22);
        v24 = *(_QWORD *)(v19 + 8LL * v22);
        if ( v18 == v24 )
          goto LABEL_11;
        ++v21;
        v20 = v23;
        v23 = (__int64 *)(v19 + 8LL * v22);
      }
      if ( !v20 )
        v20 = v23;
      v25 = v16 + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v25 < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(a1 + 20) - v25 <= v17 >> 3 )
        {
          sub_2E61F50(a1, v17);
          v61 = *(_DWORD *)(a1 + 24);
          if ( !v61 )
            goto LABEL_113;
          v24 = *(_QWORD *)(i + 32);
          v62 = v61 - 1;
          v19 = *(_QWORD *)(a1 + 8);
          v63 = 0;
          v64 = 1;
          v65 = v62 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v20 = (__int64 *)(v19 + 8LL * v65);
          v66 = *v20;
          v25 = *(_DWORD *)(a1 + 16) + 1;
          if ( v24 != *v20 )
          {
            while ( v66 != -4096 )
            {
              if ( !v63 && v66 == -8192 )
                v63 = v20;
              v65 = v62 & (v64 + v65);
              v20 = (__int64 *)(v19 + 8LL * v65);
              v66 = *v20;
              if ( v24 == *v20 )
                goto LABEL_20;
              ++v64;
            }
            if ( v63 )
              v20 = v63;
          }
        }
        goto LABEL_20;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_2E61F50(a1, 2 * v17);
    v55 = *(_DWORD *)(a1 + 24);
    if ( !v55 )
      goto LABEL_113;
    v19 = *(_QWORD *)(i + 32);
    v24 = (unsigned int)(v55 - 1);
    v56 = *(_QWORD *)(a1 + 8);
    v57 = v24 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v20 = (__int64 *)(v56 + 8LL * v57);
    v58 = *v20;
    v25 = *(_DWORD *)(a1 + 16) + 1;
    if ( v19 != *v20 )
    {
      v59 = 1;
      v60 = 0;
      while ( v58 != -4096 )
      {
        if ( v58 == -8192 && !v60 )
          v60 = v20;
        v57 = v24 & (v59 + v57);
        v20 = (__int64 *)(v56 + 8LL * v57);
        v58 = *v20;
        if ( v19 == *v20 )
          goto LABEL_20;
        ++v59;
      }
      if ( v60 )
        v20 = v60;
    }
LABEL_20:
    *(_DWORD *)(a1 + 16) = v25;
    if ( *v20 != -4096 )
      --*(_DWORD *)(a1 + 20);
    v26 = *(_QWORD *)(i + 32);
    *v20 = v26;
    v27 = *(unsigned int *)(a1 + 40);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(v67, v68, v27 + 1, 8u, v24, v19);
      v27 = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v27) = v26;
    ++*(_DWORD *)(a1 + 40);
    result = sub_220EF30(i);
  }
  return result;
}
