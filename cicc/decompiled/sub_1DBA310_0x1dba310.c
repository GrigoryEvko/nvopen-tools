// Function: sub_1DBA310
// Address: 0x1dba310
//
unsigned __int64 __fastcall sub_1DBA310(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r14
  unsigned __int64 result; // rax
  __int64 v12; // rbx
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 v23; // rcx
  unsigned __int64 j; // rdx
  __int64 v25; // rdi
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 *v28; // rax
  __int64 v29; // r11
  __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rbx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rsi
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int16 v41; // di
  __int64 k; // rax
  unsigned int v43; // ecx
  __int64 v44; // rdi
  unsigned int v45; // esi
  __int64 *v46; // rax
  __int64 v47; // r10
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned __int64 v50; // r13
  __int64 v51; // rax
  int v52; // eax
  int v53; // r10d
  int v54; // eax
  __int64 v55; // rcx
  _DWORD *v56; // rax
  _DWORD *i; // rdx
  unsigned __int64 v58; // [rsp+8h] [rbp-58h]
  __int64 v59; // [rsp+10h] [rbp-50h]
  _DWORD *v60; // [rsp+18h] [rbp-48h]
  const void *v61; // [rsp+20h] [rbp-40h]
  __int64 v62; // [rsp+28h] [rbp-38h]
  __int64 v63; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD **)(a1 + 232);
  v8 = *(unsigned int *)(a1 + 600);
  v9 = (__int64)(v7[13] - v7[12]) >> 3;
  if ( (unsigned int)v9 >= v8 )
  {
    if ( (unsigned int)v9 > v8 )
    {
      if ( (unsigned int)v9 > (unsigned __int64)*(unsigned int *)(a1 + 604) )
      {
        sub_16CD150(a1 + 592, (const void *)(a1 + 608), (unsigned int)v9, 8, a5, a6);
        v8 = *(unsigned int *)(a1 + 600);
      }
      v55 = *(_QWORD *)(a1 + 592);
      v56 = (_DWORD *)(v55 + 8 * v8);
      for ( i = (_DWORD *)(v55 + 8LL * (unsigned int)v9); i != v56; v56 += 2 )
      {
        if ( v56 )
        {
          *v56 = 0;
          v56[1] = 0;
        }
      }
      *(_DWORD *)(a1 + 600) = v9;
      v7 = *(_QWORD **)(a1 + 232);
    }
  }
  else
  {
    *(_DWORD *)(a1 + 600) = v9;
  }
  v10 = v7[41];
  result = (unsigned __int64)(v7 + 40);
  v58 = result;
  v61 = (const void *)(a1 + 528);
  if ( v10 != result )
  {
    while ( 1 )
    {
      v60 = (_DWORD *)(*(_QWORD *)(a1 + 592) + 8LL * *(int *)(v10 + 48));
      *v60 = *(_DWORD *)(a1 + 440);
      v12 = sub_1DD76D0(v10, *(_QWORD *)(a1 + 248));
      if ( v12 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 392LL) + 16LL * *(unsigned int *)(v10 + 48));
        v16 = *(unsigned int *)(a1 + 440);
        if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 444) )
        {
          sub_16CD150(a1 + 432, (const void *)(a1 + 448), 0, 8, v13, v14);
          v16 = *(unsigned int *)(a1 + 440);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 432) + 8 * v16) = v15;
        v17 = *(unsigned int *)(a1 + 520);
        ++*(_DWORD *)(a1 + 440);
        if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 524) )
        {
          sub_16CD150(a1 + 512, v61, 0, 8, v13, v14);
          v17 = *(unsigned int *)(a1 + 520);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 512) + 8 * v17) = v12;
        ++*(_DWORD *)(a1 + 520);
      }
      v18 = *(_QWORD *)(v10 + 32);
      if ( v10 + 24 != v18 )
      {
        v59 = v10;
        v19 = v10 + 24;
        while ( 1 )
        {
          v20 = *(_QWORD *)(v18 + 32);
          if ( v20 == v20 + 40LL * *(unsigned int *)(v18 + 40) )
            goto LABEL_26;
          v21 = v19;
          v22 = v20 + 40LL * *(unsigned int *)(v18 + 40);
          do
          {
            while ( *(_BYTE *)v20 != 12 )
            {
              v20 += 40;
              if ( v22 == v20 )
                goto LABEL_25;
            }
            v23 = *(_QWORD *)(a1 + 272);
            for ( j = v18; (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
              ;
            v25 = *(_QWORD *)(v23 + 368);
            v26 = *(unsigned int *)(v23 + 384);
            if ( (_DWORD)v26 )
            {
              v14 = v26 - 1;
              v27 = (v26 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
              v28 = (__int64 *)(v25 + 16LL * v27);
              v29 = *v28;
              if ( *v28 == j )
                goto LABEL_20;
              v52 = 1;
              while ( v29 != -8 )
              {
                v53 = v52 + 1;
                v27 = v14 & (v52 + v27);
                v28 = (__int64 *)(v25 + 16LL * v27);
                v29 = *v28;
                if ( *v28 == j )
                  goto LABEL_20;
                v52 = v53;
              }
            }
            v28 = (__int64 *)(v25 + 16 * v26);
LABEL_20:
            v30 = v28[1];
            v31 = *(unsigned int *)(a1 + 440);
            v32 = v30 & 0xFFFFFFFFFFFFFFF8LL | 4;
            if ( (unsigned int)v31 >= *(_DWORD *)(a1 + 444) )
            {
              v63 = v21;
              sub_16CD150(a1 + 432, (const void *)(a1 + 448), 0, 8, v21, v14);
              v31 = *(unsigned int *)(a1 + 440);
              v21 = v63;
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 432) + 8 * v31) = v32;
            v33 = *(unsigned int *)(a1 + 520);
            ++*(_DWORD *)(a1 + 440);
            v34 = *(_QWORD *)(v20 + 24);
            if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 524) )
            {
              v62 = v21;
              sub_16CD150(a1 + 512, v61, 0, 8, v21, v14);
              v33 = *(unsigned int *)(a1 + 520);
              v21 = v62;
            }
            v20 += 40;
            *(_QWORD *)(*(_QWORD *)(a1 + 512) + 8 * v33) = v34;
            ++*(_DWORD *)(a1 + 520);
          }
          while ( v22 != v20 );
LABEL_25:
          v19 = v21;
LABEL_26:
          if ( (*(_BYTE *)v18 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v18 + 46) & 8) != 0 )
              v18 = *(_QWORD *)(v18 + 8);
          }
          v18 = *(_QWORD *)(v18 + 8);
          if ( v19 == v18 )
          {
            v10 = v59;
            break;
          }
        }
      }
      v35 = sub_1DD7700(v10, *(_QWORD *)(a1 + 248));
      if ( v35 )
        break;
LABEL_44:
      result = (unsigned int)(*(_DWORD *)(a1 + 440) - *v60);
      v60[1] = result;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v58 == v10 )
        return result;
    }
    v38 = *(_QWORD *)(a1 + 272);
    v39 = *(_QWORD *)(v10 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v39 )
      BUG();
    v40 = *(_QWORD *)(v10 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v41 = *(_WORD *)(v39 + 46) & 4;
    if ( (*(_QWORD *)v39 & 4) != 0 )
    {
      if ( v41 )
      {
        do
          v39 = *(_QWORD *)v39 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v39 + 46) & 4) != 0 );
      }
    }
    else
    {
      if ( v41 )
      {
        for ( k = *(_QWORD *)v39; ; k = *(_QWORD *)v40 )
        {
          v40 = k & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v40 + 46) & 4) == 0 )
            break;
        }
      }
      v39 = v40;
    }
    v43 = *(_DWORD *)(v38 + 384);
    v44 = *(_QWORD *)(v38 + 368);
    if ( v43 )
    {
      v36 = v43 - 1;
      v45 = (v43 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v46 = (__int64 *)(v44 + 16LL * v45);
      v47 = *v46;
      if ( v39 == *v46 )
      {
LABEL_39:
        v48 = v46[1];
        v49 = *(unsigned int *)(a1 + 440);
        v50 = v48 & 0xFFFFFFFFFFFFFFF8LL | 4;
        if ( (unsigned int)v49 >= *(_DWORD *)(a1 + 444) )
        {
          sub_16CD150(a1 + 432, (const void *)(a1 + 448), 0, 8, v36, v37);
          v49 = *(unsigned int *)(a1 + 440);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 432) + 8 * v49) = v50;
        v51 = *(unsigned int *)(a1 + 520);
        ++*(_DWORD *)(a1 + 440);
        if ( (unsigned int)v51 >= *(_DWORD *)(a1 + 524) )
        {
          sub_16CD150(a1 + 512, v61, 0, 8, v36, v37);
          v51 = *(unsigned int *)(a1 + 520);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 512) + 8 * v51) = v35;
        ++*(_DWORD *)(a1 + 520);
        goto LABEL_44;
      }
      v54 = 1;
      while ( v47 != -8 )
      {
        v37 = v54 + 1;
        v45 = v36 & (v54 + v45);
        v46 = (__int64 *)(v44 + 16LL * v45);
        v47 = *v46;
        if ( v39 == *v46 )
          goto LABEL_39;
        v54 = v37;
      }
    }
    v46 = (__int64 *)(v44 + 16LL * v43);
    goto LABEL_39;
  }
  return result;
}
