// Function: sub_37B70B0
// Address: 0x37b70b0
//
_QWORD *__fastcall sub_37B70B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 *v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx

  v6 = 1;
  v8 = 1;
  *(_DWORD *)(*a1 + 8) = 0;
  v9 = a1[1];
  v10 = *a1;
  if ( *(_BYTE *)(v9 + 9) )
  {
    v6 = sub_AF4EB0(*(_QWORD *)v9);
    v8 = v6;
    if ( *(unsigned int *)(v10 + 12) >= (unsigned __int64)v6 )
      goto LABEL_3;
  }
  else if ( *(_DWORD *)(v10 + 12) )
  {
LABEL_3:
    v11 = *(unsigned int *)(v10 + 8);
    v12 = v8;
    if ( v11 <= v8 )
      v12 = *(unsigned int *)(v10 + 8);
    if ( v12 )
    {
      v13 = *(__int64 **)v10;
      v14 = *(_QWORD *)v10 + 40 * v12;
      do
      {
        v15 = *v13;
        *((_DWORD *)v13 + 2) = 0;
        v13 += 5;
        *(v13 - 3) = 0;
        *(v13 - 2) = 0;
        *(v13 - 1) = 0;
        *(v13 - 5) = v15 & 0xFFFFFFF000000000LL | 0x800000000LL;
      }
      while ( (__int64 *)v14 != v13 );
      v11 = *(unsigned int *)(v10 + 8);
    }
    if ( v11 < v8 )
    {
      v20 = *(_QWORD *)v10 + 40 * v11;
      v21 = v8 - v11;
      if ( v8 != v11 )
      {
        do
        {
          if ( v20 )
          {
            v22 = *(_QWORD *)v20;
            *(_DWORD *)(v20 + 8) = 0;
            *(_QWORD *)(v20 + 16) = 0;
            *(_QWORD *)(v20 + 24) = 0;
            *(_QWORD *)(v20 + 32) = 0;
            *(_QWORD *)v20 = v22 & 0xFFFFFFF000000000LL | 0x800000000LL;
          }
          v20 += 40;
          --v21;
        }
        while ( v21 );
      }
    }
    goto LABEL_10;
  }
  *(_DWORD *)(v10 + 8) = 0;
  sub_C8D5F0(v10, (const void *)(v10 + 16), v8, 0x28u, a5, a6);
  v17 = *(unsigned __int64 **)v10;
  v18 = *(_QWORD *)v10 + 40 * v8;
  do
  {
    if ( v17 )
    {
      *(_BYTE *)v17 = 0;
      v19 = *v17;
      *((_DWORD *)v17 + 2) = 0;
      v17[2] = 0;
      v17[3] = 0;
      *v17 = v19 & 0xFFFFFFF0000000FFLL | 0x800000000LL;
      v17[4] = 0;
    }
    v17 += 5;
  }
  while ( v17 != (unsigned __int64 *)v18 );
LABEL_10:
  *(_DWORD *)(v10 + 8) = v6;
  return sub_2E908B0(
           *(_QWORD **)a1[3],
           (unsigned __int8 **)a1[4],
           (_WORD *)a1[5],
           0,
           *(const __m128i **)*a1,
           *(unsigned int *)(*a1 + 8),
           *(_QWORD *)a1[6],
           *(_QWORD *)a1[1]);
}
