// Function: sub_39F2230
// Address: 0x39f2230
//
void __fastcall sub_39F2230(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  const void *v8; // r15
  __int64 v9; // r12
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rax
  char *v15; // rdi
  char *v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // r12
  __int64 v22; // r15
  int v23; // ecx
  __int64 v24; // rcx
  __int64 v25; // r15
  unsigned __int64 v26; // rdi
  int v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  char *v29; // [rsp+18h] [rbp-58h] BYREF
  __int64 v30; // [rsp+20h] [rbp-50h]
  _BYTE dest[72]; // [rsp+28h] [rbp-48h] BYREF

  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_QWORD *)(a1 + 264);
  v8 = *(const void **)a3;
  v29 = dest;
  v30 = 0x300000000LL;
  v9 = 8 * v6;
  if ( v6 > 3 )
  {
    sub_16CD150((__int64)&v29, dest, v6, 8, a5, a6);
    v16 = &v29[8 * (unsigned int)v30];
  }
  else
  {
    if ( !v9 )
      goto LABEL_3;
    v16 = dest;
  }
  memcpy(v16, v8, 8 * v6);
  LODWORD(v9) = v30;
LABEL_3:
  v10 = v9 + v6;
  v11 = *(unsigned int *)(v7 + 516);
  v12 = *(unsigned int *)(v7 + 512);
  LODWORD(v30) = v10;
  if ( (unsigned int)v12 >= (unsigned int)v11 )
  {
    v17 = ((((((unsigned __int64)(v11 + 2) >> 1) | (v11 + 2)) >> 2) | ((unsigned __int64)(v11 + 2) >> 1) | (v11 + 2)) >> 4)
        | ((((unsigned __int64)(v11 + 2) >> 1) | (v11 + 2)) >> 2)
        | ((unsigned __int64)(v11 + 2) >> 1)
        | (v11 + 2);
    v18 = ((v17 >> 8) | v17 | (((v17 >> 8) | v17) >> 16) | (((v17 >> 8) | v17) >> 32)) + 1;
    v19 = 0xFFFFFFFFLL;
    if ( v18 <= 0xFFFFFFFF )
      v19 = v18;
    v27 = v19;
    v13 = malloc(48 * v19);
    if ( !v13 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v12 = *(unsigned int *)(v7 + 512);
    }
    v20 = *(_QWORD *)(v7 + 504);
    v21 = v20 + 48 * v12;
    if ( v20 != v21 )
    {
      v22 = v13;
      do
      {
        if ( v22 )
        {
          v23 = *(_DWORD *)v20;
          *(_DWORD *)(v22 + 16) = 0;
          *(_DWORD *)(v22 + 20) = 3;
          *(_DWORD *)v22 = v23;
          *(_QWORD *)(v22 + 8) = v22 + 24;
          v24 = *(unsigned int *)(v20 + 16);
          if ( (_DWORD)v24 )
          {
            v28 = v20;
            sub_39F1FB0(v22 + 8, (char **)(v20 + 8), v20, v24, a5, a6);
            v20 = v28;
          }
        }
        v20 += 48;
        v22 += 48;
      }
      while ( v21 != v20 );
      v25 = *(_QWORD *)(v7 + 504);
      v21 = v25 + 48LL * *(unsigned int *)(v7 + 512);
      if ( v25 != v21 )
      {
        do
        {
          v21 -= 48LL;
          v26 = *(_QWORD *)(v21 + 8);
          if ( v26 != v21 + 24 )
            _libc_free(v26);
        }
        while ( v21 != v25 );
        v21 = *(_QWORD *)(v7 + 504);
      }
    }
    if ( v21 != v7 + 520 )
      _libc_free(v21);
    *(_QWORD *)(v7 + 504) = v13;
    LODWORD(v12) = *(_DWORD *)(v7 + 512);
    *(_DWORD *)(v7 + 516) = v27;
  }
  else
  {
    v13 = *(_QWORD *)(v7 + 504);
  }
  v14 = v13 + 48LL * (unsigned int)v12;
  if ( v14 )
  {
    *(_QWORD *)(v14 + 16) = 0x300000000LL;
    *(_DWORD *)v14 = a2;
    *(_QWORD *)(v14 + 8) = v14 + 24;
    if ( (_DWORD)v30 )
      sub_39F1FB0(v14 + 8, &v29, (unsigned int)v30, 0x300000000LL, a5, a6);
    LODWORD(v12) = *(_DWORD *)(v7 + 512);
  }
  v15 = v29;
  *(_DWORD *)(v7 + 512) = v12 + 1;
  if ( v15 != dest )
    _libc_free((unsigned __int64)v15);
}
