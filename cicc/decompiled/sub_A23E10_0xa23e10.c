// Function: sub_A23E10
// Address: 0xa23e10
//
void __fastcall sub_A23E10(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v7; // rax
  _BOOL8 v8; // r13
  unsigned __int16 v9; // ax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  __int64 *v13; // r13
  __int64 v14; // rcx
  __int64 *i; // r15
  int v16; // esi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r8
  __int64 v21; // rdx
  int v22; // esi
  __int64 v23; // rdi
  __int64 v24; // r8
  int v25; // edx
  int v26; // r9d
  __int64 v27; // [rsp+8h] [rbp-38h]

  if ( !*a4 )
    *a4 = sub_A23C00((__int64 *)a1);
  v7 = *(unsigned int *)(a3 + 8);
  v8 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v7 + 1, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  ++*(_DWORD *)(a3 + 8);
  v9 = sub_AF2710(a2);
  sub_A188E0(a3, v9);
  v10 = *(unsigned int *)(a3 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v10 + 1, 8);
    v10 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = 0;
  v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v11;
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
  {
    v13 = *(__int64 **)(a2 - 32);
    v14 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v14 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    v13 = (__int64 *)(a2 - 16 - 8LL * ((v12 >> 2) & 0xF));
  }
  for ( i = &v13[v14]; i != v13; *(_DWORD *)(a3 + 8) = v11 )
  {
    v22 = *(_DWORD *)(a1 + 304);
    v23 = *v13;
    v24 = *(_QWORD *)(a1 + 288);
    if ( v22 )
    {
      v16 = v22 - 1;
      v17 = v16 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v18 = (__int64 *)(v24 + 16LL * v17);
      v19 = *v18;
      if ( v23 == *v18 )
      {
LABEL_12:
        v20 = *((unsigned int *)v18 + 3);
        v21 = v11 + 1;
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          goto LABEL_16;
        goto LABEL_13;
      }
      v25 = 1;
      while ( v19 != -4096 )
      {
        v26 = v25 + 1;
        v17 = v16 & (v25 + v17);
        v18 = (__int64 *)(v24 + 16LL * v17);
        v19 = *v18;
        if ( v23 == *v18 )
          goto LABEL_12;
        v25 = v26;
      }
    }
    v21 = v11 + 1;
    v20 = 0;
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
LABEL_16:
      v27 = v20;
      sub_C8D5F0(a3, a3 + 16, v21, 8);
      v11 = *(unsigned int *)(a3 + 8);
      v20 = v27;
    }
LABEL_13:
    ++v13;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v20;
    v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  }
  sub_A1BFB0(*(_QWORD *)a1, 0xCu, a3, *a4);
  *(_DWORD *)(a3 + 8) = 0;
}
