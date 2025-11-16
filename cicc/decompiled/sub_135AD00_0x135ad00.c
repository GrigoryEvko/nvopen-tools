// Function: sub_135AD00
// Address: 0x135ad00
//
__int64 __fastcall sub_135AD00(_QWORD *a1)
{
  __int64 v2; // r12
  char *v3; // rax
  char *v4; // rbx
  _QWORD *v5; // rbx
  _QWORD *i; // r12
  char *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _BYTE *v12; // rdi
  __int64 *v13; // r12
  __int64 *v14; // rbx
  int v15; // eax
  int v16; // edx
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // r12
  _QWORD *v22; // [rsp+8h] [rbp-48h] BYREF
  void *src; // [rsp+10h] [rbp-40h] BYREF
  char *v24; // [rsp+18h] [rbp-38h]
  char *v25; // [rsp+20h] [rbp-30h]

  src = 0;
  v24 = 0;
  v25 = 0;
  if ( dword_4F97B80 )
  {
    v2 = 8LL * (unsigned int)dword_4F97B80;
    v3 = (char *)sub_22077B0(v2);
    v4 = v3;
    if ( v24 - (_BYTE *)src > 0 )
    {
      memmove(v3, src, v24 - (_BYTE *)src);
      j_j___libc_free_0(src, v25 - (_BYTE *)src);
    }
    src = v4;
    v24 = v4;
    v25 = &v4[v2];
  }
  v5 = (_QWORD *)a1[2];
  for ( i = a1 + 1; i != v5; v5 = (_QWORD *)v5[1] )
  {
    while ( 1 )
    {
      v22 = v5;
      v7 = v24;
      if ( v24 != v25 )
        break;
      sub_135AB70((__int64)&src, v24, &v22);
      v5 = (_QWORD *)v5[1];
      if ( i == v5 )
        goto LABEL_12;
    }
    if ( v24 )
    {
      *(_QWORD *)v24 = v5;
      v7 = v24;
    }
    v24 = v7 + 8;
  }
LABEL_12:
  v8 = sub_22077B0(72);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 16) = 0;
    v9 = 0;
    *(_QWORD *)(v8 + 24) = v8 + 16;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 64) = 0;
  }
  else
  {
    v9 = MEMORY[0] & 7;
  }
  v10 = a1[1];
  *(_QWORD *)(v8 + 8) = i;
  v10 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v8 = v10 | v9;
  *(_QWORD *)(v10 + 8) = v8;
  v11 = a1[1] & 7LL | v8;
  a1[1] = v11;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  a1[8] = v11;
  *(_BYTE *)(v11 + 67) |= 0x40u;
  *(_BYTE *)(a1[8] + 67LL) |= 0x30u;
  *(_BYTE *)(a1[8] + 67LL) |= 8u;
  v12 = src;
  v13 = (__int64 *)v24;
  if ( v24 != src )
  {
    v14 = (__int64 *)src;
    do
    {
      while ( 1 )
      {
        v17 = *v14;
        v18 = a1[8];
        v19 = *(_QWORD *)(*v14 + 32);
        if ( v19 )
          break;
        ++v14;
        sub_1357740(a1[8], v17, (__int64)a1);
        if ( v13 == v14 )
          goto LABEL_21;
      }
      *(_QWORD *)(v17 + 32) = v18;
      *(_DWORD *)(v18 + 64) = (*(_DWORD *)(v18 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v18 + 64) & 0xF8000000;
      v15 = *(_DWORD *)(v19 + 64);
      v16 = (v15 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v19 + 64) = v16 | v15 & 0xF8000000;
      if ( !v16 )
        sub_1357730(v19, (__int64)a1);
      ++v14;
    }
    while ( v13 != v14 );
LABEL_21:
    v12 = src;
  }
  v20 = a1[8];
  if ( v12 )
    j_j___libc_free_0(v12, v25 - v12);
  return v20;
}
