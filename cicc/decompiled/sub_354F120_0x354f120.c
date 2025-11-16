// Function: sub_354F120
// Address: 0x354f120
//
void __fastcall sub_354F120(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // r8
  __int64 v9; // r12
  unsigned __int64 v10; // r13
  int *v11; // rcx
  _DWORD *v12; // rdi
  __int64 v13; // rsi
  int *v14; // r9
  int *v15; // rax
  int *v16; // rdx
  unsigned int v17; // edx
  int v18; // r10d
  int *v19; // rdx
  unsigned __int64 v20; // r11
  unsigned __int64 v21; // r10
  __int64 v22; // r15
  int v23; // r14d
  int *v24; // r10
  int v25; // ecx
  unsigned __int64 v26; // r14
  int v27; // r14d
  int *v28; // rax
  int v29; // r11d
  int v30; // r10d
  __int64 v31; // rax
  int v32; // edx
  __int64 v33; // [rsp+0h] [rbp-50h]
  int *v34; // [rsp+8h] [rbp-48h]
  unsigned __int64 v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v33 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v35, a6);
  v8 = *(_QWORD *)a1;
  v9 = v7;
  v10 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == v10 )
    goto LABEL_20;
  v11 = (int *)(v7 + 80);
  v12 = (_DWORD *)(v7 + 16);
  v13 = v7;
  v14 = (int *)(v8 + 16);
  do
  {
    if ( !v13 )
      goto LABEL_14;
    *((_QWORD *)v12 - 2) = 0;
    v15 = v12;
    v16 = v12;
    *(_DWORD *)(v13 + 8) = 1;
    *(v12 - 1) = 0;
    do
    {
      if ( v16 )
        *v16 = -1;
      ++v16;
    }
    while ( v16 != v11 );
    v17 = *(_DWORD *)(v8 + 8) & 0xFFFFFFFE;
    *(_DWORD *)(v8 + 8) = *(_DWORD *)(v13 + 8) & 0xFFFFFFFE | *(_DWORD *)(v8 + 8) & 1;
    *(_DWORD *)(v13 + 8) = v17 | *(_DWORD *)(v13 + 8) & 1;
    v18 = *(v12 - 1);
    v19 = v14;
    *(v12 - 1) = *(v14 - 1);
    *(v14 - 1) = v18;
    if ( (*(_BYTE *)(v13 + 8) & 1) == 0 )
    {
      if ( (*(_BYTE *)(v8 + 8) & 1) == 0 )
      {
        v31 = *(_QWORD *)v12;
        *(_QWORD *)v12 = *(_QWORD *)v14;
        v32 = v14[2];
        *(_QWORD *)v14 = v31;
        LODWORD(v31) = v12[2];
        v12[2] = v32;
        v14[2] = v31;
        goto LABEL_14;
      }
      v19 = v12;
      v15 = v14;
      v20 = v8;
      v21 = v13;
LABEL_11:
      v22 = *(_QWORD *)(v21 + 16);
      v23 = *(_DWORD *)(v21 + 24);
      v34 = v11;
      *(_BYTE *)(v21 + 8) |= 1u;
      v24 = v15 + 16;
      do
      {
        v25 = *v15++;
        *v19++ = v25;
      }
      while ( v24 != v15 );
      *(_BYTE *)(v20 + 8) &= ~1u;
      v11 = v34;
      *(_QWORD *)(v20 + 16) = v22;
      *(_DWORD *)(v20 + 24) = v23;
      goto LABEL_14;
    }
    v20 = v13;
    v21 = v8;
    if ( (*(_BYTE *)(v8 + 8) & 1) == 0 )
      goto LABEL_11;
    v28 = v12;
    do
    {
      v29 = *v19;
      v30 = *v28++;
      ++v19;
      *(v28 - 1) = v29;
      *(v19 - 1) = v30;
    }
    while ( v11 != v28 );
LABEL_14:
    v8 += 80LL;
    v13 += 80;
    v11 += 20;
    v12 += 20;
    v14 += 20;
  }
  while ( v10 != v8 );
  v26 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    do
    {
      v10 -= 80LL;
      if ( (*(_BYTE *)(v10 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v10 + 16), 4LL * *(unsigned int *)(v10 + 24), 4);
    }
    while ( v10 != v26 );
    v10 = *(_QWORD *)a1;
  }
LABEL_20:
  v27 = v35[0];
  if ( v33 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v27;
}
