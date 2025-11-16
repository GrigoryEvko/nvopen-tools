// Function: sub_2E99DE0
// Address: 0x2e99de0
//
__int64 __fastcall sub_2E99DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r11d
  __int64 *v11; // r8
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 result; // rax
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  __int64 v20; // rcx
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // rdi
  unsigned int v24; // esi
  __int64 v25; // r8
  int v26; // r10d
  __int64 v27; // rdx
  unsigned int v28; // edi
  __int64 v29; // rcx
  unsigned int v31; // r8d
  int v32; // eax
  int v33; // ecx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  __int64 v37; // rdi
  int v38; // r10d
  __int64 v39; // r9
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // r8
  unsigned int v43; // r15d
  int v44; // r9d
  __int64 v45; // rsi
  int v46; // eax
  int v47; // ecx
  __int64 v48; // rsi
  unsigned int v49; // eax
  __int64 v50; // rdi
  int v51; // r10d
  __int64 *v52; // r9
  int v53; // eax
  int v54; // eax
  __int64 v55; // rsi
  int v56; // r9d
  unsigned int v57; // r15d
  __int64 *v58; // rdi
  __int64 v59; // rcx

  v8 = *(_DWORD *)(a3 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_59;
  }
  v9 = *(_QWORD *)(a3 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_3:
    result = *((unsigned int *)v13 + 2);
    if ( !(_DWORD)result )
      goto LABEL_18;
    return result;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_3;
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v16 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v8 )
  {
LABEL_59:
    sub_2E51FB0(a3, 2 * v8);
    v46 = *(_DWORD *)(a3 + 24);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a3 + 8);
      v49 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a3 + 16) + 1;
      v11 = (__int64 *)(v48 + 16LL * v49);
      v50 = *v11;
      if ( *v11 != a2 )
      {
        v51 = 1;
        v52 = 0;
        while ( v50 != -4096 )
        {
          if ( !v52 && v50 == -8192 )
            v52 = v11;
          v49 = v47 & (v51 + v49);
          v11 = (__int64 *)(v48 + 16LL * v49);
          v50 = *v11;
          if ( *v11 == a2 )
            goto LABEL_15;
          ++v51;
        }
        if ( v52 )
          v11 = v52;
      }
      goto LABEL_15;
    }
    goto LABEL_92;
  }
  if ( v8 - *(_DWORD *)(a3 + 20) - v17 <= v8 >> 3 )
  {
    sub_2E51FB0(a3, v8);
    v53 = *(_DWORD *)(a3 + 24);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(a3 + 8);
      v56 = 1;
      v57 = v54 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a3 + 16) + 1;
      v58 = 0;
      v11 = (__int64 *)(v55 + 16LL * v57);
      v59 = *v11;
      if ( *v11 != a2 )
      {
        while ( v59 != -4096 )
        {
          if ( v59 == -8192 && !v58 )
            v58 = v11;
          v57 = v54 & (v56 + v57);
          v11 = (__int64 *)(v55 + 16LL * v57);
          v59 = *v11;
          if ( *v11 == a2 )
            goto LABEL_15;
          ++v56;
        }
        if ( v58 )
          v11 = v58;
      }
      goto LABEL_15;
    }
LABEL_92:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a3 + 16) = v17;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v11 = a2;
  *((_DWORD *)v11 + 2) = 0;
  while ( 1 )
  {
LABEL_18:
    v18 = (unsigned int)(*(_DWORD *)(a1 + 648) - 1);
    *(_DWORD *)(a1 + 648) = v18;
    v19 = (unsigned __int64 *)(*(_QWORD *)(a1 + 640) + 48 * v18);
    if ( (unsigned __int64 *)*v19 != v19 + 2 )
      _libc_free(*v19);
    result = *(unsigned int *)(a4 + 24);
    v20 = *(_QWORD *)(a4 + 8);
    if ( !(_DWORD)result )
      return result;
    v21 = result - 1;
    v22 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v20 + 16LL * v22;
    v23 = *(_QWORD *)result;
    if ( *(_QWORD *)result != a2 )
    {
      result = 1;
      while ( v23 != -4096 )
      {
        v31 = result + 1;
        v22 = v21 & (result + v22);
        result = v20 + 16LL * v22;
        v23 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
          goto LABEL_22;
        result = v31;
      }
      return result;
    }
LABEL_22:
    a2 = *(_QWORD *)(result + 8);
    if ( !a2 )
      return result;
    v24 = *(_DWORD *)(a3 + 24);
    if ( !v24 )
    {
      ++*(_QWORD *)a3;
      goto LABEL_45;
    }
    v25 = *(_QWORD *)(a3 + 8);
    v26 = 1;
    v27 = 0;
    v28 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v25 + 16LL * v28;
    v29 = *(_QWORD *)result;
    if ( a2 != *(_QWORD *)result )
      break;
LABEL_25:
    if ( (*(_DWORD *)(result + 8))-- != 1 )
      return result;
  }
  while ( v29 != -4096 )
  {
    if ( v29 == -8192 && !v27 )
      v27 = result;
    v28 = (v24 - 1) & (v26 + v28);
    result = v25 + 16LL * v28;
    v29 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_25;
    ++v26;
  }
  if ( !v27 )
    v27 = result;
  v32 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v33 = v32 + 1;
  if ( 4 * (v32 + 1) < 3 * v24 )
  {
    result = v24 - *(_DWORD *)(a3 + 20) - v33;
    if ( (unsigned int)result > v24 >> 3 )
      goto LABEL_41;
    sub_2E51FB0(a3, v24);
    v40 = *(_DWORD *)(a3 + 24);
    if ( v40 )
    {
      result = (unsigned int)(v40 - 1);
      v41 = *(_QWORD *)(a3 + 8);
      v42 = 0;
      v43 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v44 = 1;
      v33 = *(_DWORD *)(a3 + 16) + 1;
      v27 = v41 + 16LL * v43;
      v45 = *(_QWORD *)v27;
      if ( a2 != *(_QWORD *)v27 )
      {
        while ( v45 != -4096 )
        {
          if ( !v42 && v45 == -8192 )
            v42 = v27;
          v43 = result & (v44 + v43);
          v27 = v41 + 16LL * v43;
          v45 = *(_QWORD *)v27;
          if ( a2 == *(_QWORD *)v27 )
            goto LABEL_41;
          ++v44;
        }
        if ( v42 )
          v27 = v42;
      }
      goto LABEL_41;
    }
LABEL_93:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_45:
  sub_2E51FB0(a3, 2 * v24);
  v34 = *(_DWORD *)(a3 + 24);
  if ( !v34 )
    goto LABEL_93;
  v35 = v34 - 1;
  v36 = *(_QWORD *)(a3 + 8);
  result = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v33 = *(_DWORD *)(a3 + 16) + 1;
  v27 = v36 + 16 * result;
  v37 = *(_QWORD *)v27;
  if ( a2 != *(_QWORD *)v27 )
  {
    v38 = 1;
    v39 = 0;
    while ( v37 != -4096 )
    {
      if ( !v39 && v37 == -8192 )
        v39 = v27;
      result = v35 & (unsigned int)(v38 + result);
      v27 = v36 + 16LL * (unsigned int)result;
      v37 = *(_QWORD *)v27;
      if ( a2 == *(_QWORD *)v27 )
        goto LABEL_41;
      ++v38;
    }
    if ( v39 )
      v27 = v39;
  }
LABEL_41:
  *(_DWORD *)(a3 + 16) = v33;
  if ( *(_QWORD *)v27 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *(_QWORD *)v27 = a2;
  *(_DWORD *)(v27 + 8) = -1;
  return result;
}
