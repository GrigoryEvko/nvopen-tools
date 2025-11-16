// Function: sub_33C82B0
// Address: 0x33c82b0
//
void __fastcall sub_33C82B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  const void *v8; // r15
  signed __int64 v9; // r12
  unsigned __int64 v10; // rdx
  int v11; // r12d
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  _BYTE *v14; // rsi
  _BYTE *v15; // rdi
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  _BYTE *v18; // rax
  int v19; // eax
  _BYTE *v20; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+8h] [rbp-B8h]
  _BYTE src[176]; // [rsp+10h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(const void **)(a2 + 8);
  v20 = src;
  v21 = 0x2000000000LL;
  v9 = 4 * v7;
  v10 = v9 >> 2;
  if ( (unsigned __int64)v9 > 0x80 )
  {
    sub_C8D5F0((__int64)&v20, src, v10, 4u, a5, a6);
    v17 = &v20[4 * (unsigned int)v21];
  }
  else
  {
    if ( !v9 )
    {
      LODWORD(v21) = 0;
      v11 = 0;
      goto LABEL_4;
    }
    v17 = src;
  }
  memcpy(v17, v8, v9);
  v18 = v20;
  LODWORD(v10) = (v9 >> 2) + v21;
  LODWORD(v21) = v10;
  v11 = v10;
  if ( v20 == src )
  {
LABEL_4:
    v12 = *(unsigned int *)(a3 + 8);
    v10 = (unsigned int)v10;
    if ( v12 >= (unsigned int)v10 )
    {
      v15 = src;
      if ( (_DWORD)v10 )
      {
        memmove(*(void **)a3, src, 4LL * (unsigned int)v10);
        v15 = v20;
      }
      goto LABEL_10;
    }
    if ( *(unsigned int *)(a3 + 12) < (unsigned __int64)(unsigned int)v10 )
    {
      *(_DWORD *)(a3 + 8) = 0;
      sub_C8D5F0(a3, (const void *)(a3 + 16), (unsigned int)v10, 4u, a5, a6);
      v15 = v20;
      v12 = 0;
      v16 = 4LL * (unsigned int)v21;
      v14 = v20;
      if ( v20 == &v20[v16] )
      {
LABEL_10:
        *(_DWORD *)(a3 + 8) = v11;
        if ( v15 != src )
          _libc_free((unsigned __int64)v15);
        return;
      }
    }
    else
    {
      v13 = 4 * v12;
      v14 = src;
      v15 = src;
      if ( *(_DWORD *)(a3 + 8) )
      {
        memmove(*(void **)a3, src, 4 * v12);
        v15 = v20;
        v10 = (unsigned int)v21;
        v12 = v13;
        v14 = &v20[v13];
      }
      v16 = 4 * v10;
      if ( v14 == &v15[v16] )
        goto LABEL_10;
    }
    memcpy((void *)(v12 + *(_QWORD *)a3), v14, v16 - v12);
    v15 = v20;
    goto LABEL_10;
  }
  if ( *(_QWORD *)a3 != a3 + 16 )
  {
    _libc_free(*(_QWORD *)a3);
    v18 = v20;
    v11 = v21;
  }
  *(_QWORD *)a3 = v18;
  v19 = HIDWORD(v21);
  *(_DWORD *)(a3 + 8) = v11;
  *(_DWORD *)(a3 + 12) = v19;
}
