// Function: sub_23C78F0
// Address: 0x23c78f0
//
__int64 *__fastcall sub_23C78F0(__int64 a1, const void *a2, size_t a3, int a4)
{
  unsigned int v5; // r8d
  _QWORD *v6; // rbx
  __int64 *result; // rax
  __int64 v8; // rax
  unsigned int v9; // r8d
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // [rsp+Ch] [rbp-34h]
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v5 = sub_C92740(a1, a2, a3, a4);
  v6 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
    --*(_DWORD *)(a1 + 16);
  }
  v13 = v5;
  v8 = sub_C7D670(a3 + 41, 8);
  v9 = v13;
  v10 = (_QWORD *)v8;
  if ( a3 )
  {
    memcpy((void *)(v8 + 40), a2, a3);
    v9 = v13;
  }
  v14 = v9;
  *((_BYTE *)v10 + a3 + 40) = 0;
  *v10 = a3;
  v11 = sub_22077B0(0x30u);
  if ( v11 )
  {
    *(_OWORD *)(v11 + 16) = 0;
    *(_BYTE *)(v11 + 20) = 88;
    *(_OWORD *)v11 = 0;
    *(_OWORD *)(v11 + 32) = 0;
  }
  v10[1] = v11;
  v10[2] = 0;
  v10[3] = 0;
  v10[4] = 0x2000000000LL;
  *v6 = v10;
  ++*(_DWORD *)(a1 + 12);
  result = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v14));
  v12 = *result;
  if ( *result )
    goto LABEL_11;
  do
  {
    do
    {
      v12 = result[1];
      ++result;
    }
    while ( !v12 );
LABEL_11:
    ;
  }
  while ( v12 == -8 );
  return result;
}
