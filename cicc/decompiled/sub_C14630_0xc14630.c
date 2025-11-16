// Function: sub_C14630
// Address: 0xc14630
//
__int64 __fastcall sub_C14630(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r14
  size_t *v4; // r13
  size_t v5; // r15
  const void *v6; // r13
  unsigned int v7; // eax
  unsigned int v8; // r12d
  __int64 *v9; // r9
  __int64 result; // rax
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 *v13; // r9
  __int64 v14; // rcx
  __int64 *v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 *v18; // [rsp+10h] [rbp-40h]

  v3 = a1 + 304;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v4 = *(size_t **)(a2 - 8);
    v5 = *v4;
    v6 = v4 + 3;
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  v7 = sub_C92610(v6, v5);
  v8 = sub_C92740(v3, v6, v5, v7);
  v9 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * v8);
  result = *v9;
  if ( *v9 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 320);
  }
  v18 = v9;
  v12 = sub_C7D670(v5 + 17, 8);
  v13 = v18;
  v14 = v12;
  if ( v5 )
  {
    v17 = v12;
    memcpy((void *)(v12 + 16), v6, v5);
    v13 = v18;
    v14 = v17;
  }
  *(_BYTE *)(v14 + v5 + 16) = 0;
  *(_QWORD *)v14 = v5;
  *(_DWORD *)(v14 + 8) = 0;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 316);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)sub_C929D0(v3, v8));
  result = *v15;
  if ( !*v15 || result == -8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        result = *v16++;
      while ( !result );
    }
    while ( result == -8 );
  }
LABEL_5:
  v11 = *(_DWORD *)(result + 8);
  if ( v11 > 3 )
  {
    if ( v11 == 5 )
      goto LABEL_7;
  }
  else
  {
    if ( v11 <= 1 )
    {
LABEL_7:
      *(_DWORD *)(result + 8) = 5 * (a3 == 24) + 1;
      return result;
    }
    *(_DWORD *)(result + 8) = (a3 == 24) + 3;
  }
  return result;
}
