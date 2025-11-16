// Function: sub_34E7860
// Address: 0x34e7860
//
unsigned __int64 *__fastcall sub_34E7860(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned int v12; // r8d
  int v13; // esi
  unsigned __int64 v14; // rax
  unsigned int v15; // edi
  int v16; // ecx
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r14
  __int64 v24; // r15
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdi
  unsigned __int8 v29; // cl
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+8h] [rbp-38h]

  v9 = a1;
  if ( a2 != a1 )
  {
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
        && ((v29 = *(_BYTE *)(v11 + 20), (v29 & 1) == 0) && (*(_BYTE *)(v14 + 20) & 1) != 0
         || ((*(_BYTE *)(v14 + 20) ^ v29) & 1) == 0
         && (v12 < v15
          || v12 == v15
          && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 16LL) + 24LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v14 + 16LL)
                                                                                + 24LL))) )
      {
        *a3 = 0;
        v17 = *a5;
        *a5 = v11;
        if ( v17 )
          j_j___libc_free_0(v17);
        ++a3;
        ++a5;
        if ( a2 == v9 )
          break;
      }
      else
      {
        *v9 = 0;
        v10 = *a5;
        *a5 = v14;
        if ( v10 )
          j_j___libc_free_0(v10);
        ++v9;
        ++a5;
        if ( a2 == v9 )
          break;
      }
    }
  }
  v30 = (char *)a2 - (char *)v9;
  v18 = a2 - v9;
  if ( (char *)a2 - (char *)v9 > 0 )
  {
    v19 = a5;
    do
    {
      v20 = *v9;
      *v9 = 0;
      v21 = *v19;
      *v19 = v20;
      if ( v21 )
      {
        v31 = v18;
        j_j___libc_free_0(v21);
        v18 = v31;
      }
      ++v9;
      ++v19;
      --v18;
    }
    while ( v18 );
    v22 = v30;
    if ( v30 <= 0 )
      v22 = 8;
    a5 = (unsigned __int64 *)((char *)a5 + v22);
  }
  v23 = (char *)a4 - (char *)a3;
  v24 = v23 >> 3;
  if ( v23 > 0 )
  {
    v25 = a5;
    do
    {
      v26 = *a3;
      *a3 = 0;
      v27 = *v25;
      *v25 = v26;
      if ( v27 )
        j_j___libc_free_0(v27);
      ++a3;
      ++v25;
      --v24;
    }
    while ( v24 );
    return (unsigned __int64 *)((char *)a5 + v23);
  }
  return a5;
}
