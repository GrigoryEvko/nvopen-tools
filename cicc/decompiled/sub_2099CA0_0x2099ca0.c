// Function: sub_2099CA0
// Address: 0x2099ca0
//
__int64 *__fastcall sub_2099CA0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int64 v7; // r14
  bool v8; // r8
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 *v11; // rax
  int v12; // edx
  unsigned int v13; // esi
  __int64 *v14; // r14
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 *result; // rax
  __int64 v18; // rcx
  unsigned __int8 v19; // dl
  char v20; // si
  __int64 *v21; // rax
  int v22; // edx
  unsigned int v23; // esi
  __int64 *v24; // r14
  __int64 v25; // r8
  unsigned int v26; // edi
  __int64 v27; // rcx
  __int64 v28; // rax
  int v29; // eax
  int v30; // esi
  __int64 v31; // r9
  unsigned int v32; // ecx
  int v33; // edi
  __int64 v34; // r8
  int v35; // r11d
  __int64 *v36; // r10
  int v37; // ecx
  int v38; // edi
  int v39; // eax
  int v40; // esi
  __int64 v41; // r9
  unsigned int v42; // ecx
  __int64 v43; // r8
  int v44; // r11d
  __int64 *v45; // r10
  int v46; // eax
  int v47; // ecx
  __int64 v48; // r8
  __int64 *v49; // r9
  unsigned int v50; // r15d
  int v51; // r10d
  __int64 v52; // rsi
  int v53; // r11d
  __int64 *v54; // r10
  int v55; // ecx
  int v56; // eax
  int v57; // ecx
  __int64 v58; // r8
  __int64 *v59; // r9
  unsigned int v60; // r15d
  int v61; // r10d
  __int64 v62; // rsi
  int v63; // r11d
  __int64 *v64; // r10
  int v65; // [rsp+8h] [rbp-58h]
  int v66; // [rsp+8h] [rbp-58h]
  int v67; // [rsp+8h] [rbp-58h]
  int v68; // [rsp+8h] [rbp-58h]

  v7 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v7 + 16) == 88 )
  {
    v28 = sub_157F120(*(_QWORD *)(v7 + 40));
    v7 = sub_157EBA0(v28);
  }
  if ( *(_QWORD *)(v7 + 40) != *(_QWORD *)(a2 + 40) )
  {
    v8 = sub_1642D30(v7);
    v9 = 0;
    if ( !v8 )
      goto LABEL_5;
    v19 = *(_BYTE *)(v7 + 16);
    if ( v19 <= 0x17u )
      goto LABEL_5;
    if ( v19 == 78 )
    {
      v20 = 1;
      v9 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( v19 != 29 )
        goto LABEL_5;
      v20 = 0;
      v9 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    }
    v10 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    if ( v20 )
    {
LABEL_6:
      v11 = sub_20542C0(a1, v7, **(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v10 + 48) + 24LL) + 16LL), a3, a4, a5);
      v13 = *(_DWORD *)(a1 + 32);
      v14 = v11;
      if ( v13 )
      {
        v15 = *(_QWORD *)(a1 + 16);
        v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        result = (__int64 *)(v15 + 24LL * v16);
        v18 = *result;
        if ( a2 == *result )
        {
LABEL_8:
          result[1] = (__int64)v14;
          *((_DWORD *)result + 4) = v12;
          return result;
        }
        v35 = 1;
        v36 = 0;
        while ( v18 != -8 )
        {
          if ( !v36 && v18 == -16 )
            v36 = result;
          v16 = (v13 - 1) & (v35 + v16);
          result = (__int64 *)(v15 + 24LL * v16);
          v18 = *result;
          if ( a2 == *result )
            goto LABEL_8;
          ++v35;
        }
        v37 = *(_DWORD *)(a1 + 24);
        if ( v36 )
          result = v36;
        ++*(_QWORD *)(a1 + 8);
        v38 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a1 + 28) - v38 > v13 >> 3 )
          {
LABEL_31:
            *(_DWORD *)(a1 + 24) = v38;
            if ( *result != -8 )
              --*(_DWORD *)(a1 + 28);
            *result = a2;
            result[1] = 0;
            *((_DWORD *)result + 4) = 0;
            goto LABEL_8;
          }
          v67 = v12;
          sub_205F3F0(a1 + 8, v13);
          v46 = *(_DWORD *)(a1 + 32);
          if ( v46 )
          {
            v47 = v46 - 1;
            v48 = *(_QWORD *)(a1 + 16);
            v49 = 0;
            v50 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v12 = v67;
            v51 = 1;
            v38 = *(_DWORD *)(a1 + 24) + 1;
            result = (__int64 *)(v48 + 24LL * v50);
            v52 = *result;
            if ( a2 != *result )
            {
              while ( v52 != -8 )
              {
                if ( !v49 && v52 == -16 )
                  v49 = result;
                v50 = v47 & (v51 + v50);
                result = (__int64 *)(v48 + 24LL * v50);
                v52 = *result;
                if ( a2 == *result )
                  goto LABEL_31;
                ++v51;
              }
              if ( v49 )
                result = v49;
            }
            goto LABEL_31;
          }
LABEL_96:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      v66 = v12;
      sub_205F3F0(a1 + 8, 2 * v13);
      v39 = *(_DWORD *)(a1 + 32);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 16);
        v12 = v66;
        v42 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v38 = *(_DWORD *)(a1 + 24) + 1;
        result = (__int64 *)(v41 + 24LL * v42);
        v43 = *result;
        if ( a2 != *result )
        {
          v44 = 1;
          v45 = 0;
          while ( v43 != -8 )
          {
            if ( !v45 && v43 == -16 )
              v45 = result;
            v42 = v40 & (v44 + v42);
            result = (__int64 *)(v41 + 24LL * v42);
            v43 = *result;
            if ( a2 == *result )
              goto LABEL_31;
            ++v44;
          }
          if ( v45 )
            result = v45;
        }
        goto LABEL_31;
      }
      goto LABEL_96;
    }
LABEL_5:
    v10 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    goto LABEL_6;
  }
  v21 = sub_20685E0(a1, (__int64 *)v7, a3, a4, a5);
  v23 = *(_DWORD *)(a1 + 32);
  v24 = v21;
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_20;
  }
  v25 = *(_QWORD *)(a1 + 16);
  v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v25 + 24LL * v26);
  v27 = *result;
  if ( a2 != *result )
  {
    v53 = 1;
    v54 = 0;
    while ( v27 != -8 )
    {
      if ( !v54 && v27 == -16 )
        v54 = result;
      v26 = (v23 - 1) & (v53 + v26);
      result = (__int64 *)(v25 + 24LL * v26);
      v27 = *result;
      if ( a2 == *result )
        goto LABEL_15;
      ++v53;
    }
    v55 = *(_DWORD *)(a1 + 24);
    if ( v54 )
      result = v54;
    ++*(_QWORD *)(a1 + 8);
    v33 = v55 + 1;
    if ( 4 * (v55 + 1) < 3 * v23 )
    {
      if ( v23 - *(_DWORD *)(a1 + 28) - v33 > v23 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 24) = v33;
        if ( *result != -8 )
          --*(_DWORD *)(a1 + 28);
        *result = a2;
        result[1] = 0;
        *((_DWORD *)result + 4) = 0;
        goto LABEL_15;
      }
      v68 = v22;
      sub_205F3F0(a1 + 8, v23);
      v56 = *(_DWORD *)(a1 + 32);
      if ( v56 )
      {
        v57 = v56 - 1;
        v58 = *(_QWORD *)(a1 + 16);
        v59 = 0;
        v60 = (v56 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v22 = v68;
        v61 = 1;
        v33 = *(_DWORD *)(a1 + 24) + 1;
        result = (__int64 *)(v58 + 24LL * v60);
        v62 = *result;
        if ( a2 != *result )
        {
          while ( v62 != -8 )
          {
            if ( v62 == -16 && !v59 )
              v59 = result;
            v60 = v57 & (v61 + v60);
            result = (__int64 *)(v58 + 24LL * v60);
            v62 = *result;
            if ( a2 == *result )
              goto LABEL_22;
            ++v61;
          }
          if ( v59 )
            result = v59;
        }
        goto LABEL_22;
      }
LABEL_95:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_20:
    v65 = v22;
    sub_205F3F0(a1 + 8, 2 * v23);
    v29 = *(_DWORD *)(a1 + 32);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 16);
      v22 = v65;
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = *(_DWORD *)(a1 + 24) + 1;
      result = (__int64 *)(v31 + 24LL * v32);
      v34 = *result;
      if ( a2 != *result )
      {
        v63 = 1;
        v64 = 0;
        while ( v34 != -8 )
        {
          if ( !v64 && v34 == -16 )
            v64 = result;
          v32 = v30 & (v63 + v32);
          result = (__int64 *)(v31 + 24LL * v32);
          v34 = *result;
          if ( a2 == *result )
            goto LABEL_22;
          ++v63;
        }
        if ( v64 )
          result = v64;
      }
      goto LABEL_22;
    }
    goto LABEL_95;
  }
LABEL_15:
  result[1] = (__int64)v24;
  *((_DWORD *)result + 4) = v22;
  return result;
}
