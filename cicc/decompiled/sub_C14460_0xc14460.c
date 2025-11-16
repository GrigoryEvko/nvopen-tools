// Function: sub_C14460
// Address: 0xc14460
//
__int64 __fastcall sub_C14460(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  size_t *v3; // r13
  size_t v4; // r12
  const void *v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r8d
  __int64 *v8; // rcx
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 *v12; // rcx
  __int64 v13; // r15
  __int64 *v14; // rdx
  __int64 *v15; // rdx
  __int64 *v16; // [rsp+0h] [rbp-40h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

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
  result = *v8;
  if ( *v8 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 320);
  }
  v16 = v8;
  v17 = v7;
  v10 = sub_C7D670(v4 + 17, 8);
  v11 = v17;
  v12 = v16;
  v13 = v10;
  if ( v4 )
  {
    memcpy((void *)(v10 + 16), v5, v4);
    v11 = v17;
    v12 = v16;
  }
  *(_BYTE *)(v13 + v4 + 16) = 0;
  *(_QWORD *)v13 = v4;
  *(_DWORD *)(v13 + 8) = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 316);
  v14 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)sub_C929D0(v2, v11));
  result = *v14;
  if ( !*v14 || result == -8 )
  {
    v15 = v14 + 1;
    do
    {
      do
        result = *v15++;
      while ( !result );
    }
    while ( result == -8 );
  }
LABEL_5:
  switch ( *(_DWORD *)(result + 8) )
  {
    case 0:
    case 2:
    case 5:
      *(_DWORD *)(result + 8) = 2;
      break;
    case 1:
    case 3:
      *(_DWORD *)(result + 8) = 3;
      break;
    case 6:
      *(_DWORD *)(result + 8) = 4;
      break;
    default:
      return result;
  }
  return result;
}
