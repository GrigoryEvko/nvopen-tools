// Function: sub_DAE3E0
// Address: 0xdae3e0
//
__int64 __fastcall sub_DAE3E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned int v7; // esi
  int v8; // r10d
  __int64 v9; // r8
  _QWORD *v10; // r15
  unsigned int v11; // ecx
  __int64 *v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // r9
  __int64 *v15; // rdx
  unsigned __int64 **v16; // r15
  unsigned __int64 *v17; // rsi
  int v18; // r8d
  __int64 result; // rax
  int v20; // edi
  int v21; // ecx
  unsigned int v22; // esi
  int v23; // r11d
  __int64 v24; // r9
  unsigned int v25; // r8d
  __int64 *v26; // rcx
  _QWORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 i; // rdx
  int v31; // eax
  int v32; // edx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r8d
  _QWORD *v37; // r9
  int v38; // edx
  int v39; // edx
  int v40; // r8d
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rsi
  int v44; // ebx
  int v45; // edi
  int v46; // edx
  int v47; // r9d
  __int64 v48; // r10
  unsigned int v49; // ecx
  __int64 v50; // r8
  int v51; // esi
  _QWORD *v52; // r11
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // rax
  int v55; // edx
  int v56; // r8d
  __int64 v57; // r10
  _QWORD *v58; // r9
  int v59; // ecx
  __int64 v60; // r15
  __int64 v61; // rsi
  unsigned int v62; // [rsp+8h] [rbp-38h]
  unsigned int v63; // [rsp+8h] [rbp-38h]
  unsigned int v64; // [rsp+8h] [rbp-38h]

  v3 = a1 + 904;
  v7 = *(_DWORD *)(a1 + 928);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 904);
    goto LABEL_32;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 912);
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v12 = (__int64 *)(v9 + 40LL * v11);
  v13 = *v12;
  if ( *v12 != a2 )
  {
    while ( v13 != -4096 )
    {
      if ( !v10 && v13 == -8192 )
        v10 = v12;
      v11 = (v7 - 1) & (v8 + v11);
      v12 = (__int64 *)(v9 + 40LL * v11);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v8;
    }
    if ( !v10 )
      v10 = v12;
    v20 = *(_DWORD *)(a1 + 920);
    ++*(_QWORD *)(a1 + 904);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 924) - v21 > v7 >> 3 )
      {
LABEL_19:
        *(_DWORD *)(a1 + 920) = v21;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 924);
        v17 = v10 + 3;
        *v10 = a2;
        v16 = (unsigned __int64 **)(v10 + 1);
        *v16 = v17;
        v16[1] = (unsigned __int64 *)0x200000000LL;
        goto LABEL_22;
      }
      v62 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_DAE0C0(v3, v7);
      v38 = *(_DWORD *)(a1 + 928);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = 1;
        v37 = 0;
        v41 = *(_QWORD *)(a1 + 912);
        LODWORD(v42) = v39 & v62;
        v10 = (_QWORD *)(v41 + 40LL * (v39 & v62));
        v43 = *v10;
        v21 = *(_DWORD *)(a1 + 920) + 1;
        if ( *v10 == a2 )
          goto LABEL_19;
        while ( v43 != -4096 )
        {
          if ( v43 == -8192 && !v37 )
            v37 = v10;
          v42 = v39 & (unsigned int)(v42 + v40);
          v10 = (_QWORD *)(v41 + 40 * v42);
          v43 = *v10;
          if ( *v10 == a2 )
            goto LABEL_19;
          ++v40;
        }
        goto LABEL_36;
      }
      goto LABEL_95;
    }
LABEL_32:
    sub_DAE0C0(v3, 2 * v7);
    v31 = *(_DWORD *)(a1 + 928);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 912);
      v34 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (_QWORD *)(v33 + 40LL * v34);
      v35 = *v10;
      v21 = *(_DWORD *)(a1 + 920) + 1;
      if ( *v10 == a2 )
        goto LABEL_19;
      v36 = 1;
      v37 = 0;
      while ( v35 != -4096 )
      {
        if ( !v37 && v35 == -8192 )
          v37 = v10;
        v34 = v32 & (v36 + v34);
        v10 = (_QWORD *)(v33 + 40LL * v34);
        v35 = *v10;
        if ( *v10 == a2 )
          goto LABEL_19;
        ++v36;
      }
LABEL_36:
      if ( v37 )
        v10 = v37;
      goto LABEL_19;
    }
LABEL_95:
    ++*(_DWORD *)(a1 + 920);
    BUG();
  }
LABEL_3:
  v14 = *((unsigned int *)v12 + 4);
  v15 = (__int64 *)v12[1];
  v16 = (unsigned __int64 **)(v12 + 1);
  v17 = (unsigned __int64 *)&v15[v14];
  v18 = v14;
  if ( v15 == (__int64 *)v17 )
  {
LABEL_68:
    v53 = *((unsigned int *)v12 + 5);
    if ( v53 <= v14 )
    {
      v54 = a3 & 0xFFFFFFFFFFFFFFF9LL;
      if ( v14 + 1 > v53 )
      {
        sub_C8D5F0((__int64)(v12 + 1), v12 + 3, v14 + 1, 8u, v14, v14);
        v54 = a3 & 0xFFFFFFFFFFFFFFF9LL;
        v17 = (unsigned __int64 *)(v12[1] + 8LL * *((unsigned int *)v12 + 4));
      }
      *v17 = v54;
      ++*((_DWORD *)v12 + 4);
LABEL_24:
      result = sub_DAEA10(a1, a2, a3);
      v22 = *(_DWORD *)(a1 + 928);
      if ( v22 )
      {
        v23 = 1;
        v24 = *(_QWORD *)(a1 + 912);
        v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v26 = (__int64 *)(v24 + 40LL * v25);
        v27 = 0;
        v28 = *v26;
        if ( *v26 == a2 )
        {
LABEL_26:
          v29 = v26[1];
          for ( i = v29 + 8LL * *((unsigned int *)v26 + 4); i != v29; i -= 8 )
          {
            if ( a3 == (*(_QWORD *)(i - 8) & 0xFFFFFFFFFFFFFFF8LL) )
            {
              *(_QWORD *)(i - 8) = (2LL * (unsigned int)result) | *(_QWORD *)(i - 8) & 0xFFFFFFFFFFFFFFF9LL;
              return result;
            }
          }
          return result;
        }
        while ( v28 != -4096 )
        {
          if ( !v27 && v28 == -8192 )
            v27 = v26;
          v25 = (v22 - 1) & (v23 + v25);
          v26 = (__int64 *)(v24 + 40LL * v25);
          v28 = *v26;
          if ( *v26 == a2 )
            goto LABEL_26;
          ++v23;
        }
        v44 = *(_DWORD *)(a1 + 920);
        if ( !v27 )
          v27 = v26;
        ++*(_QWORD *)(a1 + 904);
        v45 = v44 + 1;
        if ( 4 * (v44 + 1) < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a1 + 924) - v45 > v22 >> 3 )
          {
LABEL_57:
            *(_DWORD *)(a1 + 920) = v45;
            if ( *v27 != -4096 )
              --*(_DWORD *)(a1 + 924);
            *v27 = a2;
            v27[1] = v27 + 3;
            v27[2] = 0x200000000LL;
            return result;
          }
          v64 = result;
          sub_DAE0C0(v3, v22);
          v55 = *(_DWORD *)(a1 + 928);
          if ( v55 )
          {
            v56 = v55 - 1;
            v57 = *(_QWORD *)(a1 + 912);
            v58 = 0;
            v59 = 1;
            LODWORD(v60) = (v55 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v45 = *(_DWORD *)(a1 + 920) + 1;
            result = v64;
            v27 = (_QWORD *)(v57 + 40LL * (unsigned int)v60);
            v61 = *v27;
            if ( *v27 != a2 )
            {
              while ( v61 != -4096 )
              {
                if ( !v58 && v61 == -8192 )
                  v58 = v27;
                v60 = v56 & (unsigned int)(v60 + v59);
                v27 = (_QWORD *)(v57 + 40 * v60);
                v61 = *v27;
                if ( *v27 == a2 )
                  goto LABEL_57;
                ++v59;
              }
              if ( v58 )
                v27 = v58;
            }
            goto LABEL_57;
          }
LABEL_96:
          ++*(_DWORD *)(a1 + 920);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 904);
      }
      v63 = result;
      sub_DAE0C0(v3, 2 * v22);
      v46 = *(_DWORD *)(a1 + 928);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 912);
        v49 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v45 = *(_DWORD *)(a1 + 920) + 1;
        result = v63;
        v27 = (_QWORD *)(v48 + 40LL * v49);
        v50 = *v27;
        if ( *v27 != a2 )
        {
          v51 = 1;
          v52 = 0;
          while ( v50 != -4096 )
          {
            if ( v50 == -8192 && !v52 )
              v52 = v27;
            v49 = v47 & (v51 + v49);
            v27 = (_QWORD *)(v48 + 40LL * v49);
            v50 = *v27;
            if ( *v27 == a2 )
              goto LABEL_57;
            ++v51;
          }
          if ( v52 )
            v27 = v52;
        }
        goto LABEL_57;
      }
      goto LABEL_96;
    }
    if ( !v17 )
    {
LABEL_23:
      *((_DWORD *)v16 + 2) = v18 + 1;
      goto LABEL_24;
    }
LABEL_22:
    *v17 = a3 & 0xFFFFFFFFFFFFFFF9LL;
    v18 = *((_DWORD *)v16 + 2);
    goto LABEL_23;
  }
  while ( a3 != (*v15 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( v17 == (unsigned __int64 *)++v15 )
      goto LABEL_68;
  }
  return (*v15 >> 1) & 3;
}
