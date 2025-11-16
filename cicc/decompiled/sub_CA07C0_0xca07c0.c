// Function: sub_CA07C0
// Address: 0xca07c0
//
__int64 *__fastcall sub_CA07C0(__int64 a1, const void *a2, size_t a3, int a4)
{
  unsigned int v5; // r8d
  _QWORD *v6; // rbx
  __int64 *result; // rax
  __int64 v8; // rax
  unsigned int v9; // r8d
  _QWORD *v10; // r13
  __int64 v11; // rdx
  unsigned int v12; // [rsp+Ch] [rbp-34h]

  v5 = sub_C92740(a1, a2, a3, a4);
  v6 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
    --*(_DWORD *)(a1 + 16);
  }
  v12 = v5;
  v8 = sub_C7D670(a3 + 185, 8);
  v9 = v12;
  v10 = (_QWORD *)v8;
  if ( a3 )
  {
    memcpy((void *)(v8 + 184), a2, a3);
    v9 = v12;
  }
  *((_BYTE *)v10 + a3 + 184) = 0;
  *v10 = a3;
  memset(v10 + 1, 0, 0xB0u);
  v10[11] = v10 + 13;
  v10[15] = v10 + 17;
  *v6 = v10;
  ++*(_DWORD *)(a1 + 12);
  result = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v9));
  v11 = *result;
  if ( *result )
    goto LABEL_9;
  do
  {
    do
    {
      v11 = result[1];
      ++result;
    }
    while ( !v11 );
LABEL_9:
    ;
  }
  while ( v11 == -8 );
  return result;
}
