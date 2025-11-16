// Function: sub_FEC3C0
// Address: 0xfec3c0
//
__int64 *__fastcall sub_FEC3C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 *result; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // r12d
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  __int64 *v12; // rcx
  unsigned int v13; // r8d
  __int64 *v14; // rax
  __int64 v15; // rdi
  int v16; // eax
  _BYTE *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 *v20; // rdi
  int v21; // r11d
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  _QWORD *v26; // rdx
  _BYTE *v27; // r12
  unsigned int v28; // esi
  int v29; // ecx
  __int64 v30; // rsi
  int v31; // ecx
  __int64 v32; // r9
  unsigned int v33; // edx
  int v34; // eax
  __int64 v35; // r8
  __int64 v36; // rax
  int v37; // eax
  int v38; // ecx
  __int64 v39; // rsi
  int v40; // ecx
  __int64 v41; // r9
  __int64 *v42; // r10
  int v43; // r11d
  unsigned int v44; // edx
  __int64 v45; // r8
  int v46; // eax
  int v47; // edi
  int v48; // eax
  int v49; // edx
  __int64 v50; // rsi
  unsigned int v51; // eax
  __int64 v52; // r8
  int v53; // r10d
  __int64 *v54; // r9
  int v55; // eax
  int v56; // eax
  __int64 v57; // r8
  int v58; // r10d
  unsigned int v59; // edx
  __int64 v60; // rsi
  int v61; // r11d
  unsigned int v62; // [rsp+Ch] [rbp-34h]

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  while ( 1 )
  {
    result = *(__int64 **)(a1 + 96);
    if ( *(__int64 **)(a1 + 88) == result )
      return result;
    sub_FEC280(a1);
    v5 = *(_QWORD *)(a1 + 96);
    v6 = *(_QWORD *)(v5 - 48);
    v7 = *(_DWORD *)(v5 - 8);
    v8 = v5 - 48;
    *(_QWORD *)(a1 + 96) = v8;
    if ( *(_QWORD *)(a1 + 88) != v8 && *(_DWORD *)(v8 - 8) > v7 )
      *(_DWORD *)(v8 - 8) = v7;
    v9 = *(_DWORD *)(a1 + 32);
    if ( !v9 )
    {
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_57;
    }
    v10 = *(_QWORD *)(a1 + 16);
    v11 = 1;
    v12 = 0;
    v13 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v6 != *v14 )
    {
      while ( v15 != -4096 )
      {
        if ( !v12 && v15 == -8192 )
          v12 = v14;
        v13 = (v9 - 1) & (v11 + v13);
        v14 = (__int64 *)(v10 + 16LL * v13);
        v15 = *v14;
        if ( v6 == *v14 )
          goto LABEL_9;
        ++v11;
      }
      if ( !v12 )
        v12 = v14;
      v46 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v47 = v46 + 1;
      if ( 4 * (v46 + 1) >= 3 * v9 )
      {
LABEL_57:
        sub_FEBD00(a1 + 8, 2 * v9);
        v48 = *(_DWORD *)(a1 + 32);
        if ( !v48 )
          goto LABEL_89;
        v49 = v48 - 1;
        v50 = *(_QWORD *)(a1 + 16);
        v51 = (v48 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v47 = *(_DWORD *)(a1 + 24) + 1;
        v12 = (__int64 *)(v50 + 16LL * v51);
        v52 = *v12;
        if ( v6 != *v12 )
        {
          v53 = 1;
          v54 = 0;
          while ( v52 != -4096 )
          {
            if ( !v54 && v52 == -8192 )
              v54 = v12;
            v51 = v49 & (v53 + v51);
            v12 = (__int64 *)(v50 + 16LL * v51);
            v52 = *v12;
            if ( v6 == *v12 )
              goto LABEL_53;
            ++v53;
          }
LABEL_61:
          if ( v54 )
            v12 = v54;
        }
      }
      else if ( v9 - *(_DWORD *)(a1 + 28) - v47 <= v9 >> 3 )
      {
        v62 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
        sub_FEBD00(a1 + 8, v9);
        v55 = *(_DWORD *)(a1 + 32);
        if ( !v55 )
        {
LABEL_89:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
        v56 = v55 - 1;
        v57 = *(_QWORD *)(a1 + 16);
        v58 = 1;
        v54 = 0;
        v59 = v56 & v62;
        v47 = *(_DWORD *)(a1 + 24) + 1;
        v12 = (__int64 *)(v57 + 16LL * (v56 & v62));
        v60 = *v12;
        if ( v6 != *v12 )
        {
          while ( v60 != -4096 )
          {
            if ( v60 == -8192 && !v54 )
              v54 = v12;
            v59 = v56 & (v58 + v59);
            v12 = (__int64 *)(v57 + 16LL * v59);
            v60 = *v12;
            if ( v6 == *v12 )
              goto LABEL_53;
            ++v58;
          }
          goto LABEL_61;
        }
      }
LABEL_53:
      *(_DWORD *)(a1 + 24) = v47;
      if ( *v12 != -4096 )
        --*(_DWORD *)(a1 + 28);
      *v12 = v6;
      v16 = 0;
      *((_DWORD *)v12 + 2) = 0;
      goto LABEL_10;
    }
LABEL_9:
    v16 = *((_DWORD *)v14 + 2);
LABEL_10:
    if ( v7 == v16 )
    {
      v17 = *(_BYTE **)(a1 + 72);
      while ( 1 )
      {
        v25 = *(_QWORD *)(a1 + 48);
        v26 = (_QWORD *)(v25 - 8);
        if ( *(_BYTE **)(a1 + 80) == v17 )
        {
          sub_FE9C50(v1, v17, v26);
          v27 = *(_BYTE **)(a1 + 72);
          v26 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
        }
        else
        {
          if ( v17 )
          {
            *(_QWORD *)v17 = *(_QWORD *)(v25 - 8);
            v17 = *(_BYTE **)(a1 + 72);
            v26 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          v27 = v17 + 8;
          *(_QWORD *)(a1 + 72) = v17 + 8;
        }
        v28 = *(_DWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 48) = v26;
        if ( !v28 )
          break;
        v18 = *((_QWORD *)v27 - 1);
        v19 = *(_QWORD *)(a1 + 16);
        v20 = 0;
        v21 = 1;
        v22 = (v28 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v23 = (__int64 *)(v19 + 16LL * v22);
        v24 = *v23;
        if ( *v23 == v18 )
        {
LABEL_13:
          *((_DWORD *)v23 + 2) = -1;
          v17 = *(_BYTE **)(a1 + 72);
          result = v23 + 1;
          if ( v6 == *((_QWORD *)v17 - 1) )
            return result;
        }
        else
        {
          while ( v24 != -4096 )
          {
            if ( !v20 && v24 == -8192 )
              v20 = v23;
            v22 = (v28 - 1) & (v21 + v22);
            v23 = (__int64 *)(v19 + 16LL * v22);
            v24 = *v23;
            if ( v18 == *v23 )
              goto LABEL_13;
            ++v21;
          }
          if ( !v20 )
            v20 = v23;
          v37 = *(_DWORD *)(a1 + 24);
          ++*(_QWORD *)(a1 + 8);
          v34 = v37 + 1;
          if ( 4 * v34 < 3 * v28 )
          {
            if ( v28 - *(_DWORD *)(a1 + 28) - v34 > v28 >> 3 )
              goto LABEL_22;
            sub_FEBD00(a1 + 8, v28);
            v38 = *(_DWORD *)(a1 + 32);
            if ( !v38 )
              goto LABEL_89;
            v39 = *((_QWORD *)v27 - 1);
            v40 = v38 - 1;
            v41 = *(_QWORD *)(a1 + 16);
            v42 = 0;
            v43 = 1;
            v44 = v40 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v34 = *(_DWORD *)(a1 + 24) + 1;
            v20 = (__int64 *)(v41 + 16LL * v44);
            v45 = *v20;
            if ( *v20 == v39 )
              goto LABEL_22;
            while ( v45 != -4096 )
            {
              if ( v45 == -8192 && !v42 )
                v42 = v20;
              v44 = v40 & (v43 + v44);
              v20 = (__int64 *)(v41 + 16LL * v44);
              v45 = *v20;
              if ( v39 == *v20 )
                goto LABEL_22;
              ++v43;
            }
            goto LABEL_39;
          }
LABEL_20:
          sub_FEBD00(a1 + 8, 2 * v28);
          v29 = *(_DWORD *)(a1 + 32);
          if ( !v29 )
            goto LABEL_89;
          v30 = *((_QWORD *)v27 - 1);
          v31 = v29 - 1;
          v32 = *(_QWORD *)(a1 + 16);
          v33 = v31 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v34 = *(_DWORD *)(a1 + 24) + 1;
          v20 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v20;
          if ( *v20 == v30 )
            goto LABEL_22;
          v61 = 1;
          v42 = 0;
          while ( v35 != -4096 )
          {
            if ( !v42 && v35 == -8192 )
              v42 = v20;
            v33 = v31 & (v61 + v33);
            v20 = (__int64 *)(v32 + 16LL * v33);
            v35 = *v20;
            if ( v30 == *v20 )
              goto LABEL_22;
            ++v61;
          }
LABEL_39:
          if ( v42 )
            v20 = v42;
LABEL_22:
          *(_DWORD *)(a1 + 24) = v34;
          if ( *v20 != -4096 )
            --*(_DWORD *)(a1 + 28);
          v36 = *((_QWORD *)v27 - 1);
          *((_DWORD *)v20 + 2) = 0;
          *v20 = v36;
          result = v20 + 1;
          *((_DWORD *)v20 + 2) = -1;
          v17 = *(_BYTE **)(a1 + 72);
          if ( v6 == *((_QWORD *)v17 - 1) )
            return result;
        }
      }
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_20;
    }
  }
}
