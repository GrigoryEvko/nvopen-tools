// Function: sub_C147D0
// Address: 0xc147d0
//
__int64 __fastcall sub_C147D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  size_t *v3; // r13
  size_t v4; // r12
  const void *v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r8d
  __int64 *v8; // rcx
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 *v13; // rcx
  __int64 v14; // r15
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // [rsp+0h] [rbp-40h]
  unsigned int v18; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 304;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v3 = *(size_t **)(a2 - 8);
    v4 = *v3;
    v5 = v3 + 3;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = sub_C92610(v5, v4);
  v7 = sub_C92740(v2, v5, v4, v6);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * v7);
  v9 = *v8;
  if ( *v8 )
  {
    if ( v9 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 320);
  }
  v17 = v8;
  v18 = v7;
  v11 = sub_C7D670(v4 + 17, 8);
  v12 = v18;
  v13 = v17;
  v14 = v11;
  if ( v4 )
  {
    memcpy((void *)(v11 + 16), v5, v4);
    v12 = v18;
    v13 = v17;
  }
  *(_BYTE *)(v14 + v4 + 16) = 0;
  *(_QWORD *)v14 = v4;
  *(_DWORD *)(v14 + 8) = 0;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 316);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)sub_C929D0(v2, v12));
  v9 = *v15;
  if ( !*v15 || v9 == -8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        v9 = *v16++;
      while ( !v9 );
    }
    while ( v9 == -8 );
    result = *(unsigned int *)(v9 + 8);
    if ( !(_DWORD)result )
      goto LABEL_17;
    goto LABEL_6;
  }
LABEL_5:
  result = *(unsigned int *)(v9 + 8);
  if ( !(_DWORD)result )
  {
LABEL_17:
    *(_DWORD *)(v9 + 8) = 5;
    return result;
  }
LABEL_6:
  if ( (_DWORD)result == 5 )
    goto LABEL_17;
  return result;
}
