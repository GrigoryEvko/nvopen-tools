// Function: sub_342B420
// Address: 0x342b420
//
__int64 __fastcall sub_342B420(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 result; // rax
  unsigned int v8; // ecx
  unsigned int v9; // edi
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r8
  int v13; // r11d
  _QWORD *v14; // rdi
  unsigned int v15; // r15d
  unsigned int v16; // r10d
  _QWORD *v17; // rdx
  __int64 v18; // r9
  _QWORD *v19; // rdx
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rbx
  int v24; // r11d
  _QWORD *v25; // r8
  unsigned int v26; // r14d
  unsigned int v27; // r9d
  _QWORD *v28; // rdx
  __int64 v29; // rdi
  _QWORD *v30; // rdx
  int v31; // edx
  int v32; // r9d
  int v33; // edx
  int v34; // r9d
  int v35; // eax
  __int64 v36; // rdi
  int v37; // ecx
  int v38; // eax
  int v39; // edx
  int v40; // eax
  int v41; // eax
  __int64 v42; // rcx
  unsigned int v43; // r15d
  __int64 v44; // rsi
  int v45; // r10d
  _QWORD *v46; // r9
  int v47; // eax
  int v48; // eax
  __int64 v49; // rsi
  unsigned int v50; // r14d
  __int64 v51; // rdx
  int v52; // r9d
  _QWORD *v53; // rdi
  int v54; // eax
  int v55; // eax
  int v56; // r10d
  unsigned int v57; // r15d
  __int64 v58; // rcx
  __int64 v59; // rsi
  int v60; // eax
  int v61; // eax
  __int64 v62; // rsi
  int v63; // r9d
  unsigned int v64; // r14d
  __int64 v65; // rdx
  __int64 v66; // [rsp+8h] [rbp-38h]
  __int64 v67; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 16);
  v4 = **(_QWORD **)(a1 + 24);
  v5 = *(_DWORD *)(v3 + 752);
  if ( v4 )
    v4 -= 8;
  result = *(_QWORD *)(v3 + 736);
  if ( !v5 )
    return result;
  v8 = v5 - 1;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(result + 80LL * v9);
  v11 = *v10;
  if ( v4 != *v10 )
  {
    v31 = 1;
    while ( v11 != -4096 )
    {
      v32 = v31 + 1;
      v9 = v8 & (v31 + v9);
      v10 = (__int64 *)(result + 80LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        goto LABEL_5;
      v31 = v32;
    }
    goto LABEL_10;
  }
LABEL_5:
  if ( v10 != (__int64 *)(result + 80LL * v5) )
  {
    v12 = v10[5];
    if ( !v12 )
      goto LABEL_12;
    v13 = 1;
    v14 = 0;
    v15 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v16 = v15 & v8;
    v17 = (_QWORD *)(result + 80LL * (v15 & v8));
    v18 = *v17;
    if ( a2 == *v17 )
    {
LABEL_8:
      v19 = v17 + 1;
LABEL_9:
      v19[4] = v12;
      v3 = *(_QWORD *)(a1 + 16);
      result = *(_QWORD *)(v3 + 736);
      v5 = *(_DWORD *)(v3 + 752);
      goto LABEL_10;
    }
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v14 )
        v14 = v17;
      v16 = v8 & (v13 + v16);
      v17 = (_QWORD *)(result + 80LL * v16);
      v18 = *v17;
      if ( a2 == *v17 )
        goto LABEL_8;
      ++v13;
    }
    v38 = *(_DWORD *)(v3 + 744);
    if ( !v14 )
      v14 = v17;
    ++*(_QWORD *)(v3 + 728);
    v39 = v38 + 1;
    if ( 4 * (v38 + 1) >= 3 * v5 )
    {
      v66 = v12;
      sub_337D430(v3 + 728, 2 * v5);
      v40 = *(_DWORD *)(v3 + 752);
      if ( v40 )
      {
        v41 = v40 - 1;
        v42 = *(_QWORD *)(v3 + 736);
        v12 = v66;
        v43 = v41 & v15;
        v14 = (_QWORD *)(v42 + 80LL * v43);
        v39 = *(_DWORD *)(v3 + 744) + 1;
        v44 = *v14;
        if ( a2 == *v14 )
          goto LABEL_50;
        v45 = 1;
        v46 = 0;
        while ( v44 != -4096 )
        {
          if ( !v46 && v44 == -8192 )
            v46 = v14;
          v43 = v41 & (v45 + v43);
          v14 = (_QWORD *)(v42 + 80LL * v43);
          v44 = *v14;
          if ( a2 == *v14 )
            goto LABEL_50;
          ++v45;
        }
LABEL_57:
        if ( v46 )
          v14 = v46;
        goto LABEL_50;
      }
    }
    else
    {
      if ( v5 - *(_DWORD *)(v3 + 748) - v39 > v5 >> 3 )
      {
LABEL_50:
        *(_DWORD *)(v3 + 744) = v39;
        if ( *v14 != -4096 )
          --*(_DWORD *)(v3 + 748);
        *v14 = a2;
        v19 = v14 + 1;
        v14[1] = v14 + 3;
        v14[9] = 0;
        v14[2] = 0x100000000LL;
        *(_OWORD *)(v14 + 3) = 0;
        *(_OWORD *)(v14 + 5) = 0;
        *(_OWORD *)(v14 + 7) = 0;
        goto LABEL_9;
      }
      v67 = v12;
      sub_337D430(v3 + 728, v5);
      v54 = *(_DWORD *)(v3 + 752);
      if ( v54 )
      {
        v55 = v54 - 1;
        v46 = 0;
        v12 = v67;
        v56 = 1;
        v57 = v55 & v15;
        v58 = *(_QWORD *)(v3 + 736);
        v14 = (_QWORD *)(v58 + 80LL * v57);
        v39 = *(_DWORD *)(v3 + 744) + 1;
        v59 = *v14;
        if ( a2 == *v14 )
          goto LABEL_50;
        while ( v59 != -4096 )
        {
          if ( !v46 && v59 == -8192 )
            v46 = v14;
          v57 = v55 & (v56 + v57);
          v14 = (_QWORD *)(v58 + 80LL * v57);
          v59 = *v14;
          if ( a2 == *v14 )
            goto LABEL_50;
          ++v56;
        }
        goto LABEL_57;
      }
    }
    ++*(_DWORD *)(v3 + 744);
    BUG();
  }
LABEL_10:
  if ( !v5 )
    return result;
  v8 = v5 - 1;
LABEL_12:
  v20 = v8 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v21 = (__int64 *)(result + 80LL * v20);
  v22 = *v21;
  if ( v4 == *v21 )
  {
LABEL_13:
    if ( v21 != (__int64 *)(result + 80LL * v5) )
    {
      v23 = v21[6];
      if ( v23 )
      {
        v24 = 1;
        v25 = 0;
        v26 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        v27 = v26 & v8;
        v28 = (_QWORD *)(result + 80LL * (v26 & v8));
        v29 = *v28;
        if ( a2 == *v28 )
        {
LABEL_16:
          v30 = v28 + 1;
LABEL_17:
          v30[5] = v23;
          return result;
        }
        while ( v29 != -4096 )
        {
          if ( !v25 && v29 == -8192 )
            v25 = v28;
          v27 = v8 & (v24 + v27);
          v28 = (_QWORD *)(result + 80LL * v27);
          v29 = *v28;
          if ( a2 == *v28 )
            goto LABEL_16;
          ++v24;
        }
        v35 = *(_DWORD *)(v3 + 744);
        v36 = v3 + 728;
        if ( !v25 )
          v25 = v28;
        ++*(_QWORD *)(v3 + 728);
        v37 = v35 + 1;
        if ( 4 * (v35 + 1) >= 3 * v5 )
        {
          sub_337D430(v36, 2 * v5);
          v47 = *(_DWORD *)(v3 + 752);
          if ( v47 )
          {
            v48 = v47 - 1;
            v49 = *(_QWORD *)(v3 + 736);
            v50 = v48 & v26;
            v37 = *(_DWORD *)(v3 + 744) + 1;
            v25 = (_QWORD *)(v49 + 80LL * v50);
            v51 = *v25;
            if ( a2 == *v25 )
              goto LABEL_37;
            v52 = 1;
            v53 = 0;
            while ( v51 != -4096 )
            {
              if ( !v53 && v51 == -8192 )
                v53 = v25;
              v50 = v48 & (v52 + v50);
              v25 = (_QWORD *)(v49 + 80LL * v50);
              v51 = *v25;
              if ( a2 == *v25 )
                goto LABEL_37;
              ++v52;
            }
LABEL_64:
            if ( v53 )
              v25 = v53;
            goto LABEL_37;
          }
        }
        else
        {
          if ( v5 - *(_DWORD *)(v3 + 748) - v37 > v5 >> 3 )
          {
LABEL_37:
            *(_DWORD *)(v3 + 744) = v37;
            if ( *v25 != -4096 )
              --*(_DWORD *)(v3 + 748);
            *v25 = a2;
            v30 = v25 + 1;
            v25[1] = v25 + 3;
            result = 0x100000000LL;
            v25[9] = 0;
            v25[2] = 0x100000000LL;
            *(_OWORD *)(v25 + 3) = 0;
            *(_OWORD *)(v25 + 5) = 0;
            *(_OWORD *)(v25 + 7) = 0;
            goto LABEL_17;
          }
          sub_337D430(v36, v5);
          v60 = *(_DWORD *)(v3 + 752);
          if ( v60 )
          {
            v61 = v60 - 1;
            v62 = *(_QWORD *)(v3 + 736);
            v63 = 1;
            v53 = 0;
            v64 = v61 & v26;
            v37 = *(_DWORD *)(v3 + 744) + 1;
            v25 = (_QWORD *)(v62 + 80LL * v64);
            v65 = *v25;
            if ( a2 == *v25 )
              goto LABEL_37;
            while ( v65 != -4096 )
            {
              if ( v65 == -8192 && !v53 )
                v53 = v25;
              v64 = v61 & (v63 + v64);
              v25 = (_QWORD *)(v62 + 80LL * v64);
              v65 = *v25;
              if ( a2 == *v25 )
                goto LABEL_37;
              ++v63;
            }
            goto LABEL_64;
          }
        }
        ++*(_DWORD *)(v3 + 744);
        BUG();
      }
    }
  }
  else
  {
    v33 = 1;
    while ( v22 != -4096 )
    {
      v34 = v33 + 1;
      v20 = v8 & (v20 + v33);
      v21 = (__int64 *)(result + 80LL * v20);
      v22 = *v21;
      if ( v4 == *v21 )
        goto LABEL_13;
      v33 = v34;
    }
  }
  return result;
}
