// Function: sub_35D5F90
// Address: 0x35d5f90
//
__int64 __fastcall sub_35D5F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r10
  __int64 result; // rax
  int v15; // edx
  unsigned int v16; // esi
  __int64 v17; // r8
  unsigned __int64 *v18; // r10
  int v19; // r15d
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // rcx
  int v23; // r11d
  int v24; // edi
  int v25; // ecx
  int v26; // edx
  int v27; // edx
  __int64 v28; // r8
  unsigned int v29; // esi
  unsigned __int64 v30; // rdi
  int v31; // r11d
  unsigned __int64 *v32; // r9
  int v33; // edx
  int v34; // esi
  __int64 v35; // rdi
  unsigned __int64 *v36; // r8
  unsigned int v37; // r14d
  int v38; // r9d
  unsigned __int64 v39; // rdx
  unsigned int v40; // [rsp+Ch] [rbp-34h]
  unsigned int v41; // [rsp+Ch] [rbp-34h]

  v8 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = *(unsigned int *)(a1 + 120);
  v10 = *(_QWORD *)(a1 + 104);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (v8 ^ (v8 >> 9));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_3:
      if ( v12 != (__int64 *)(v10 + 16 * v9) )
        return *((unsigned int *)v12 + 2);
    }
    else
    {
      v15 = 1;
      while ( v13 != -4 )
      {
        v23 = v15 + 1;
        v11 = (v9 - 1) & (v15 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v8 == *v12 )
          goto LABEL_3;
        v15 = v23;
      }
    }
  }
  result = sub_35D5240(a1, a3, a4);
  v16 = *(_DWORD *)(a1 + 120);
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_26;
  }
  v17 = *(_QWORD *)(a1 + 104);
  v18 = 0;
  v19 = 1;
  v20 = (v16 - 1) & (v8 ^ (v8 >> 9));
  v21 = (__int64 *)(v17 + 16LL * v20);
  v22 = *v21;
  if ( v8 == *v21 )
  {
LABEL_9:
    *((_DWORD *)v21 + 2) = result;
    return result;
  }
  while ( v22 != -4 )
  {
    if ( v22 == -16 && !v18 )
      v18 = (unsigned __int64 *)v21;
    v20 = (v16 - 1) & (v19 + v20);
    v21 = (__int64 *)(v17 + 16LL * v20);
    v22 = *v21;
    if ( v8 == *v21 )
      goto LABEL_9;
    ++v19;
  }
  v24 = *(_DWORD *)(a1 + 112);
  if ( !v18 )
    v18 = (unsigned __int64 *)v21;
  ++*(_QWORD *)(a1 + 96);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v16 )
  {
LABEL_26:
    v40 = result;
    sub_35D59E0(a1 + 96, 2 * v16);
    v26 = *(_DWORD *)(a1 + 120);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 104);
      v25 = *(_DWORD *)(a1 + 112) + 1;
      result = v40;
      v29 = v27 & (v8 ^ (v8 >> 9));
      v18 = (unsigned __int64 *)(v28 + 16LL * v29);
      v30 = *v18;
      if ( v8 != *v18 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4 )
        {
          if ( v30 == -16 && !v32 )
            v32 = v18;
          v29 = v27 & (v31 + v29);
          v18 = (unsigned __int64 *)(v28 + 16LL * v29);
          v30 = *v18;
          if ( v8 == *v18 )
            goto LABEL_22;
          ++v31;
        }
        if ( v32 )
          v18 = v32;
      }
      goto LABEL_22;
    }
    goto LABEL_49;
  }
  if ( v16 - *(_DWORD *)(a1 + 116) - v25 <= v16 >> 3 )
  {
    v41 = result;
    sub_35D59E0(a1 + 96, v16);
    v33 = *(_DWORD *)(a1 + 120);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 104);
      v36 = 0;
      v37 = (v33 - 1) & (v8 ^ (v8 >> 9));
      v38 = 1;
      v25 = *(_DWORD *)(a1 + 112) + 1;
      result = v41;
      v18 = (unsigned __int64 *)(v35 + 16LL * v37);
      v39 = *v18;
      if ( v8 != *v18 )
      {
        while ( v39 != -4 )
        {
          if ( !v36 && v39 == -16 )
            v36 = v18;
          v37 = v34 & (v38 + v37);
          v18 = (unsigned __int64 *)(v35 + 16LL * v37);
          v39 = *v18;
          if ( v8 == *v18 )
            goto LABEL_22;
          ++v38;
        }
        if ( v36 )
          v18 = v36;
      }
      goto LABEL_22;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_22:
  *(_DWORD *)(a1 + 112) = v25;
  if ( *v18 != -4 )
    --*(_DWORD *)(a1 + 116);
  *v18 = v8;
  *((_DWORD *)v18 + 2) = 0;
  *((_DWORD *)v18 + 2) = result;
  return result;
}
