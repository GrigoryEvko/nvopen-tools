// Function: sub_184ACC0
// Address: 0x184acc0
//
void __fastcall sub_184ACC0(__int64 a1, __int64 a2)
{
  void *v2; // r14
  _QWORD *v3; // rax
  __int64 v5; // r12
  char v7; // al
  char v8; // dl
  unsigned __int64 v9; // rax
  unsigned int v10; // r15d
  size_t v11; // r8
  int v12; // edx
  const void *v13; // rsi
  int v14; // r9d
  void *v15; // rdi
  unsigned int v16; // r14d
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  size_t v21; // rdx

  v2 = (void *)(a1 + 16);
  v3 = (_QWORD *)(a1 + 16);
  v5 = a1 + 80;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  do
  {
    if ( v3 )
      *v3 = -8;
    ++v3;
  }
  while ( v3 != (_QWORD *)v5 );
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 16));
  v7 = *(_BYTE *)(a2 + 8);
  v8 = *(_BYTE *)(a1 + 8) | 1;
  *(_BYTE *)(a1 + 8) = v8;
  LODWORD(v9) = v7 & 1;
  if ( (_DWORD)v9 )
  {
    v11 = 64;
    v20 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFFELL | *(_QWORD *)(a1 + 8) & 1LL;
    *(_QWORD *)(a1 + 8) = v20;
    v12 = v20 & 1;
    if ( v12 )
      goto LABEL_11;
  }
  else
  {
    v10 = *(_DWORD *)(a2 + 24);
    if ( v10 > 8 )
    {
      *(_BYTE *)(a1 + 8) = v8 & 0xFE;
      v9 = sub_22077B0(8LL * v10);
      v17 = *(_QWORD *)(a1 + 8);
      v18 = *(_QWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 24) = v10;
      *(_QWORD *)(a1 + 16) = v9;
      v11 = 64;
      v19 = v18 & 0xFFFFFFFFFFFFFFFELL | v17 & 1;
      LOBYTE(v9) = *(_BYTE *)(a2 + 8) & 1;
      *(_QWORD *)(a1 + 8) = v19;
      v12 = v19 & 1;
      if ( v12 )
      {
        if ( (_BYTE)v9 )
          goto LABEL_11;
        goto LABEL_18;
      }
    }
    else
    {
      v11 = 64;
      v9 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFFELL | *(_QWORD *)(a1 + 8) & 1LL;
      *(_QWORD *)(a1 + 8) = v9;
      LODWORD(v9) = v9 & 1;
      LOBYTE(v12) = v9;
      if ( (_DWORD)v9 )
        goto LABEL_18;
    }
  }
  LOBYTE(v12) = 0;
  v11 = 8LL * *(unsigned int *)(a1 + 24);
  if ( (_BYTE)v9 )
  {
LABEL_11:
    v13 = (const void *)(a2 + 16);
    goto LABEL_12;
  }
LABEL_18:
  v13 = *(const void **)(a2 + 16);
LABEL_12:
  if ( !(_BYTE)v12 )
    v2 = *(void **)(a1 + 16);
  memcpy(v2, v13, v11);
  v15 = (void *)(a1 + 96);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x800000000LL;
  v16 = *(_DWORD *)(a2 + 88);
  if ( v16 && v5 != a2 + 80 )
  {
    v21 = 8LL * v16;
    if ( v16 <= 8
      || (sub_16CD150(v5, (const void *)(a1 + 96), v16, 8, v16, v14),
          v15 = *(void **)(a1 + 80),
          (v21 = 8LL * *(unsigned int *)(a2 + 88)) != 0) )
    {
      memcpy(v15, *(const void **)(a2 + 80), v21);
    }
    *(_DWORD *)(a1 + 88) = v16;
  }
}
