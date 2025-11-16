// Function: sub_2DF5F90
// Address: 0x2df5f90
//
unsigned __int64 __fastcall sub_2DF5F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v7; // rdi
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r10
  int v11; // r13d
  __int64 v12; // r15
  unsigned __int64 v13; // rbx
  unsigned int v14; // esi
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r11
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // r10
  unsigned __int64 v21; // rbx
  unsigned int v22; // r14d
  unsigned int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r13
  unsigned __int64 *v29; // r9
  __int64 v30; // [rsp+0h] [rbp-50h]
  unsigned __int64 v31; // [rsp+8h] [rbp-48h]
  _QWORD *v32; // [rsp+10h] [rbp-40h]

  v7 = a1 + 8;
  LODWORD(v8) = *(_DWORD *)(v7 + 8);
  v9 = *(_QWORD *)v7 + 16LL * (unsigned int)(v8 - 1);
  v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 12));
  v11 = *(_DWORD *)(*(_QWORD *)(v7 - 8) + 160LL) - v8;
  if ( v11 )
  {
    a5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = (a2 >> 1) & 3;
    do
    {
      v13 = v10 & 0xFFFFFFFFFFFFFFC0LL;
      v14 = v12 | *(_DWORD *)(a5 + 24);
      if ( (*(_DWORD *)((*(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 0x60) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 0x60) >> 1) & 3) > v14 )
      {
        v18 = (_QWORD *)(v10 & 0xFFFFFFFFFFFFFFC0LL);
        v16 = 0;
      }
      else
      {
        v15 = 0;
        do
        {
          v16 = ++v15;
          v17 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * v15 + 0x60);
        }
        while ( (*(_DWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v17 >> 1) & 3) <= v14 );
        v18 = (_QWORD *)(v13 + 8LL * v15);
      }
      v8 = (unsigned int)v8;
      v19 = (unsigned int)v8 + 1LL;
      v20 = (v16 << 32) | ((v10 & 0x3F) + 1);
      if ( v19 > *(unsigned int *)(a1 + 20) )
      {
        v30 = v20;
        v31 = a5;
        v32 = v18;
        sub_C8D5F0(v7, (const void *)(a1 + 24), v19, 0x10u, a5, (unsigned int)v8);
        v8 = *(unsigned int *)(a1 + 16);
        v20 = v30;
        a5 = v31;
        v18 = v32;
      }
      v8 = *(_QWORD *)(a1 + 8) + 16 * v8;
      *(_QWORD *)v8 = v13;
      *(_QWORD *)(v8 + 8) = v20;
      LODWORD(v8) = *(_DWORD *)(a1 + 16) + 1;
      *(_DWORD *)(a1 + 16) = v8;
      v10 = *v18;
      --v11;
    }
    while ( v11 );
  }
  v21 = v10 & 0xFFFFFFFFFFFFFFC0LL;
  v22 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  if ( v22 < (*(_DWORD *)((*(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 8) >> 1) & 3) )
  {
    v24 = 0;
  }
  else
  {
    v23 = 0;
    do
    {
      v24 = ++v23;
      v25 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * v23 + 8);
    }
    while ( (*(_DWORD *)((v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v25 >> 1) & 3) <= v22 );
  }
  v8 = (unsigned int)v8;
  result = *(unsigned int *)(a1 + 20);
  v27 = (unsigned int)v8 + 1LL;
  v28 = (v24 << 32) | ((v10 & 0x3F) + 1);
  if ( v27 > result )
  {
    result = sub_C8D5F0(v7, (const void *)(a1 + 24), v27, 0x10u, a5, (unsigned int)v8);
    v8 = *(unsigned int *)(a1 + 16);
  }
  v29 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16 * v8);
  *v29 = v21;
  v29[1] = v28;
  ++*(_DWORD *)(a1 + 16);
  return result;
}
