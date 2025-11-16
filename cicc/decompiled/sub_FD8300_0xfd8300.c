// Function: sub_FD8300
// Address: 0xfd8300
//
__int64 __fastcall sub_FD8300(__int64 a1)
{
  __int64 v2; // r12
  char *v3; // rax
  char *v4; // rbx
  __int64 v5; // rbx
  __int64 i; // r12
  char *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _BYTE *v13; // rdi
  __int64 *v14; // r12
  __int64 *v15; // rbx
  int v16; // eax
  int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // rdi
  __int64 v21; // r12
  __int64 v23; // [rsp+8h] [rbp-48h] BYREF
  void *src; // [rsp+10h] [rbp-40h] BYREF
  char *v25; // [rsp+18h] [rbp-38h]
  char *v26; // [rsp+20h] [rbp-30h]

  src = 0;
  v25 = 0;
  v26 = 0;
  if ( (_DWORD)qword_4F8D968 )
  {
    v2 = 8LL * (unsigned int)qword_4F8D968;
    v3 = (char *)sub_22077B0(v2);
    v4 = v3;
    if ( v25 - (_BYTE *)src > 0 )
    {
      memmove(v3, src, v25 - (_BYTE *)src);
      j_j___libc_free_0(src, v26 - (_BYTE *)src);
    }
    src = v4;
    v25 = v4;
    v26 = &v4[v2];
  }
  v5 = *(_QWORD *)(a1 + 16);
  for ( i = a1 + 8; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    while ( 1 )
    {
      v23 = v5;
      v7 = v25;
      if ( v25 != v26 )
        break;
      sub_FD8170((__int64)&src, v25, &v23);
      v5 = *(_QWORD *)(v5 + 8);
      if ( i == v5 )
        goto LABEL_12;
    }
    if ( v25 )
    {
      *(_QWORD *)v25 = v5;
      v7 = v25;
    }
    v25 = v7 + 8;
  }
LABEL_12:
  v8 = sub_22077B0(72);
  if ( v8 )
  {
    *(_DWORD *)(v8 + 64) &= 0x80000000;
    v10 = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = v8 + 40;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 0;
  }
  else
  {
    v10 = MEMORY[0] & 7;
  }
  v11 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v8 + 8) = i;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v8 = v11 | v10;
  *(_QWORD *)(v11 + 8) = v8;
  v12 = *(_QWORD *)(a1 + 8) & 7LL | v8;
  *(_QWORD *)(a1 + 8) = v12;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a1 + 64) = v12;
  *(_BYTE *)(v12 + 67) |= 0x40u;
  *(_BYTE *)(*(_QWORD *)(a1 + 64) + 67LL) |= 0x30u;
  *(_BYTE *)(*(_QWORD *)(a1 + 64) + 67LL) |= 8u;
  v13 = src;
  v14 = (__int64 *)v25;
  if ( v25 != src )
  {
    v15 = (__int64 *)src;
    do
    {
      while ( 1 )
      {
        v18 = *v15;
        v19 = *(_QWORD *)(a1 + 64);
        v20 = *(_QWORD *)(*v15 + 16);
        if ( v20 )
          break;
        ++v15;
        sub_FD7340(*(_QWORD *)(a1 + 64), v18, a1, *(__int64 **)a1, v19, v9);
        if ( v14 == v15 )
          goto LABEL_21;
      }
      *(_QWORD *)(v18 + 16) = v19;
      *(_DWORD *)(v19 + 64) = (*(_DWORD *)(v19 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v19 + 64) & 0xF8000000;
      v16 = *(_DWORD *)(v20 + 64);
      v17 = (v16 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v20 + 64) = v17 | v16 & 0xF8000000;
      if ( !v17 )
        sub_FD59A0(v20, a1);
      ++v15;
    }
    while ( v14 != v15 );
LABEL_21:
    v13 = src;
  }
  v21 = *(_QWORD *)(a1 + 64);
  if ( v13 )
    j_j___libc_free_0(v13, v26 - v13);
  return v21;
}
