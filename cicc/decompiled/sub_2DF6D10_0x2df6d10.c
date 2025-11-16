// Function: sub_2DF6D10
// Address: 0x2df6d10
//
unsigned __int64 *__fastcall sub_2DF6D10(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // r14d
  unsigned int v4; // r12d
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdx
  unsigned __int8 v10; // di
  __int64 v11; // rsi
  char v12; // al
  __int64 v13; // rdi
  char v14; // al
  unsigned int v15; // r12d
  unsigned __int64 *result; // rax
  __int64 v17; // rax
  const void **v18; // rsi
  void *v19; // r9
  unsigned __int64 v20; // rdi
  size_t v21; // rdx
  unsigned __int64 *v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 160LL) )
    return sub_2DF6980(a1, 1);
  v3 = *(_DWORD *)(v2 + 164);
  v4 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4) + 1;
  if ( v3 != v4 )
  {
    do
    {
      v5 = v4 - 1;
      v6 = (_QWORD *)(v2 + 16LL * v4);
      v7 = (_QWORD *)(v2 + 16 * v5);
      *v7 = *v6;
      v7[1] = v6[1];
      v8 = 24LL * v4;
      v9 = (_QWORD *)(v2 + 24 * v5 + 64);
      if ( (_QWORD *)(v2 + v8 + 64) != v9 )
      {
        if ( (*(_BYTE *)(v2 + v8 + 72) & 0x3F) != 0 )
        {
          v22 = (unsigned __int64 *)(v2 + 24 * v5 + 64);
          v24 = v2 + v8 + 64;
          v17 = sub_2207820(4LL * (*(_BYTE *)(v2 + v8 + 72) & 0x3F));
          v18 = (const void **)v24;
          v5 = v4 - 1;
          v19 = (void *)v17;
          v20 = *v22;
          *v22 = v17;
          if ( v20 )
          {
            j_j___libc_free_0_0(v20);
            v5 = v4 - 1;
            v18 = (const void **)v24;
            v19 = (void *)*v22;
          }
          v10 = *(_BYTE *)(v2 + 24LL * v4 + 72) & 0x3F;
          v21 = 4LL * v10;
          if ( v21 )
          {
            v23 = v5;
            memmove(v19, *v18, v21);
            v5 = v23;
            v10 = *(_BYTE *)(v2 + 24LL * v4 + 72) & 0x3F;
          }
        }
        else
        {
          *v9 = 0;
          v10 = *(_BYTE *)(v2 + v8 + 72) & 0x3F;
        }
        v11 = v2 + 24 * v5;
        v12 = v10 | *(_BYTE *)(v11 + 72) & 0xC0;
        v13 = v2 + 24LL * v4;
        *(_BYTE *)(v11 + 72) = v12;
        v14 = *(_BYTE *)(v13 + 72) & 0x40 | v12 & 0xBF;
        *(_BYTE *)(v11 + 72) = v14;
        *(_BYTE *)(v11 + 72) = *(_BYTE *)(v13 + 72) & 0x80 | v14 & 0x7F;
        *(_QWORD *)(v11 + 80) = *(_QWORD *)(v13 + 80);
      }
      ++v4;
    }
    while ( v3 != v4 );
    v4 = *(_DWORD *)(v2 + 164);
  }
  v15 = v4 - 1;
  *(_DWORD *)(v2 + 164) = v15;
  result = *(unsigned __int64 **)(a1 + 8);
  *((_DWORD *)result + 2) = v15;
  return result;
}
