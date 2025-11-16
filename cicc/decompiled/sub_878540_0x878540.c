// Function: sub_878540
// Address: 0x878540
//
__int64 __fastcall sub_878540(char *src, size_t n, __int64 *a3)
{
  char *v4; // rsi
  char *v5; // rax
  unsigned int v6; // r12d
  signed int v7; // r13d
  __int64 v8; // rbx
  _QWORD *v9; // r13
  __int64 result; // rax
  void *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]

  if ( n )
  {
    v4 = &src[n];
    v5 = src;
    v6 = 0;
    do
      v6 = 73 * v6 + (unsigned __int8)*v5++;
    while ( v5 != v4 );
    v7 = v6 % 0x3FFF5;
  }
  else
  {
    v7 = 0;
    v6 = 0;
  }
  v13 = v7;
  v8 = qword_4D04BE0[v7];
  if ( v8 )
  {
    v9 = 0;
    while ( *(_QWORD *)(v8 + 16) != n || memcmp(src, *(const void **)(v8 + 8), n) )
    {
      v9 = (_QWORD *)v8;
      if ( !*(_QWORD *)v8 )
        goto LABEL_13;
      v8 = *(_QWORD *)v8;
    }
    result = *(_QWORD *)(v8 + 24);
    if ( v9 )
    {
      *v9 = *(_QWORD *)v8;
      *(_QWORD *)v8 = qword_4D04BE0[v13];
      qword_4D04BE0[v13] = v8;
    }
  }
  else
  {
LABEL_13:
    v8 = sub_877070();
    *(_QWORD *)v8 = qword_4D04BE0[v13];
    qword_4D04BE0[v13] = v8;
    v11 = (void *)sub_7279A0(n + 1);
    v12 = memcpy(v11, src, n);
    v12[n] = 0;
    *(_BYTE *)(v8 + 73) &= ~1u;
    *(_QWORD *)(v8 + 16) = n;
    *(_DWORD *)(v8 + 56) = v6;
    *(_QWORD *)(v8 + 8) = v12;
    result = 0;
  }
  *a3 = v8;
  return result;
}
