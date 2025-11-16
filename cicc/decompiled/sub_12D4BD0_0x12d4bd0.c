// Function: sub_12D4BD0
// Address: 0x12d4bd0
//
char __fastcall sub_12D4BD0(int **a1, __int64 a2)
{
  int *v3; // r14
  int v4; // r13d
  __int64 v5; // rbx
  __int64 v6; // rax
  size_t v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r8
  size_t v10; // r12
  char result; // al
  int v12; // eax
  int v13; // ecx
  size_t v14; // r8
  const void *v15; // rdi
  size_t v16; // rdx
  int v17; // r10d
  unsigned int i; // r9d
  const void *v19; // rsi
  unsigned int v20; // r9d
  size_t v21; // [rsp+0h] [rbp-60h]
  int v22; // [rsp+14h] [rbp-4Ch]
  size_t v23; // [rsp+18h] [rbp-48h]
  unsigned int v24; // [rsp+18h] [rbp-48h]
  void *s1a; // [rsp+20h] [rbp-40h]
  int s1; // [rsp+20h] [rbp-40h]
  size_t na; // [rsp+28h] [rbp-38h]
  size_t n; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v4 = **a1;
  v5 = *((_QWORD *)v3 + 1) + 16LL * (unsigned int)v3[6];
  v6 = sub_1649960(a2);
  v8 = (unsigned int)v3[6];
  v9 = *((_QWORD *)v3 + 1);
  if ( (_DWORD)v8 )
  {
    v23 = *((_QWORD *)v3 + 1);
    na = v7;
    s1a = (void *)v6;
    v12 = sub_16D3930(v6, v7);
    v13 = v8 - 1;
    v14 = v23;
    v15 = s1a;
    v16 = na;
    v17 = 1;
    for ( i = (v8 - 1) & v12; ; i = v13 & v20 )
    {
      v10 = v14 + 16LL * i;
      v19 = *(const void **)v10;
      if ( *(_QWORD *)v10 == -1 )
        break;
      if ( v19 == (const void *)-2LL )
      {
        if ( v15 == (const void *)-2LL )
        {
          result = v5 != v10;
          if ( *(_BYTE *)(a2 + 16) != 3 )
            return result;
          goto LABEL_12;
        }
      }
      else if ( *(_QWORD *)(v10 + 8) == v16 )
      {
        v22 = v17;
        v24 = i;
        s1 = v13;
        n = v14;
        if ( !v16 )
          goto LABEL_3;
        v21 = v16;
        if ( !memcmp(v15, v19, v16) )
          goto LABEL_3;
        v16 = v21;
        v14 = n;
        v13 = s1;
        i = v24;
        v17 = v22;
      }
      v20 = v17 + i;
      ++v17;
    }
    if ( v15 == (const void *)-1LL )
      goto LABEL_3;
    v9 = *((_QWORD *)v3 + 1);
    v8 = (unsigned int)v3[6];
  }
  v10 = v9 + 16 * v8;
LABEL_3:
  result = v5 != v10;
  if ( *(_BYTE *)(a2 + 16) == 3 )
  {
LABEL_12:
    if ( (*(_BYTE *)(a2 + 32) & 0xF) != 0 )
    {
      if ( (*(_BYTE *)(a2 + 32) & 0xF) == 6 && v5 == v10 && v4 )
        return 0;
    }
    else if ( v5 == v10 )
    {
      return v4 == 0;
    }
    return 1;
  }
  return result;
}
