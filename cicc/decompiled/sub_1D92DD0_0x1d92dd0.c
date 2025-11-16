// Function: sub_1D92DD0
// Address: 0x1d92dd0
//
char *__fastcall sub_1D92DD0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 *v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rcx
  unsigned int v12; // r8d
  int v13; // edi
  __int64 v14; // rdx
  unsigned int v15; // eax
  int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 *v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int8 v24; // si
  __int64 v25; // r14
  __int64 v26; // r15
  __int64 *v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdi

  if ( a1 == a2 )
  {
LABEL_16:
    v18 = (char *)a4 - (char *)a3;
    v19 = v18 >> 3;
    if ( v18 > 0 )
    {
      v20 = a5;
      do
      {
        v21 = *a3;
        *a3 = 0;
        v22 = *v20;
        *v20 = v21;
        if ( v22 )
          j_j___libc_free_0(v22, 24);
        ++a3;
        ++v20;
        --v19;
      }
      while ( v19 );
      return (char *)a5 + v18;
    }
    return (char *)a5;
  }
  v9 = a1;
  while ( a4 != a3 )
  {
    v11 = *a3;
    v12 = *(_DWORD *)(*a3 + 8);
    v13 = *(_DWORD *)(*a3 + 12);
    if ( v12 == 7 )
      v13 = -(*(_DWORD *)(v11 + 16) + v13);
    v14 = *v9;
    v15 = *(_DWORD *)(*v9 + 8);
    v16 = *(_DWORD *)(*v9 + 12);
    if ( v15 == 7 )
      v16 = -(*(_DWORD *)(v14 + 16) + v16);
    if ( v13 > v16
      || v13 == v16
      && ((v24 = *(_BYTE *)(v11 + 20), (v24 & 1) == 0) && (*(_BYTE *)(v14 + 20) & 1) != 0
       || ((*(_BYTE *)(v14 + 20) ^ v24) & 1) == 0
       && (v12 < v15
        || v12 == v15
        && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 16LL) + 48LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v14 + 16LL) + 48LL))) )
    {
      *a3 = 0;
      v17 = *a5;
      *a5 = v11;
      if ( v17 )
        j_j___libc_free_0(v17, 24);
      ++a3;
      ++a5;
      if ( v9 == a2 )
        goto LABEL_16;
    }
    else
    {
      *v9 = 0;
      v10 = *a5;
      *a5 = v14;
      if ( v10 )
        j_j___libc_free_0(v10, 24);
      ++v9;
      ++a5;
      if ( v9 == a2 )
        goto LABEL_16;
    }
  }
  v25 = (char *)a2 - (char *)v9;
  v26 = v25 >> 3;
  if ( v25 <= 0 )
    return (char *)a5;
  v27 = a5;
  do
  {
    v28 = *v9;
    *v9 = 0;
    v29 = *v27;
    *v27 = v28;
    if ( v29 )
      j_j___libc_free_0(v29, 24);
    ++v9;
    ++v27;
    --v26;
  }
  while ( v26 );
  return (char *)a5 + v25;
}
