// Function: sub_A1E080
// Address: 0xa1e080
//
void __fastcall sub_A1E080(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // r8d
  __int64 v8; // rax
  _BOOL8 v9; // r12
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  __int64 *v12; // r12
  __int64 v13; // rcx
  __int64 *i; // r14
  int v15; // esi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r11
  __int64 v19; // r9
  __int64 v20; // rdx
  int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // r9
  __int64 v24; // r12
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  int v28; // edx
  int v29; // r10d
  unsigned int v30; // [rsp+4h] [rbp-3Ch]
  __int64 v31; // [rsp+8h] [rbp-38h]
  unsigned int v33; // [rsp+8h] [rbp-38h]
  unsigned int v34; // [rsp+8h] [rbp-38h]

  v4 = a4;
  v8 = *(unsigned int *)(a3 + 8);
  v9 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v8 + 1, 8);
    v8 = *(unsigned int *)(a3 + 8);
    v4 = a4;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
  v10 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v10;
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) != 0 )
  {
    v12 = *(__int64 **)(a2 - 32);
    v13 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v13 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    v12 = (__int64 *)(a2 - 16 - 8LL * ((v11 >> 2) & 0xF));
  }
  for ( i = &v12[v13]; i != v12; *(_DWORD *)(a3 + 8) = v10 )
  {
    v21 = *(_DWORD *)(a1 + 304);
    v22 = *v12;
    v23 = *(_QWORD *)(a1 + 288);
    if ( v21 )
    {
      v15 = v21 - 1;
      v16 = v15 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v17 = (__int64 *)(v23 + 16LL * v16);
      v18 = *v17;
      if ( v22 == *v17 )
      {
LABEL_8:
        v19 = *((unsigned int *)v17 + 3);
        v20 = v10 + 1;
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          goto LABEL_12;
        goto LABEL_9;
      }
      v28 = 1;
      while ( v18 != -4096 )
      {
        v29 = v28 + 1;
        v16 = v15 & (v28 + v16);
        v17 = (__int64 *)(v23 + 16LL * v16);
        v18 = *v17;
        if ( v22 == *v17 )
          goto LABEL_8;
        v28 = v29;
      }
    }
    v20 = v10 + 1;
    v19 = 0;
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
LABEL_12:
      v30 = v4;
      v31 = v19;
      sub_C8D5F0(a3, a3 + 16, v20, 8);
      v10 = *(unsigned int *)(a3 + 8);
      v4 = v30;
      v19 = v31;
    }
LABEL_9:
    ++v12;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v19;
    v10 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  }
  v24 = *(unsigned int *)(a2 + 4);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v33 = v4;
    sub_C8D5F0(a3, a3 + 16, v10 + 1, 8);
    v10 = *(unsigned int *)(a3 + 8);
    v4 = v33;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v24;
  v25 = *(unsigned int *)(a3 + 12);
  v26 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v26;
  v27 = (unsigned __int64)*(char *)(a2 + 1) >> 63;
  if ( v26 + 1 > v25 )
  {
    v34 = v4;
    sub_C8D5F0(a3, a3 + 16, v26 + 1, 8);
    v26 = *(unsigned int *)(a3 + 8);
    v4 = v34;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v26) = v27;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*(_QWORD *)a1, 0x20u, a3, v4);
  *(_DWORD *)(a3 + 8) = 0;
}
