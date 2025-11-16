// Function: sub_271E5B0
// Address: 0x271e5b0
//
__int64 __fastcall sub_271E5B0(__int64 a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rcx
  __int64 result; // rax
  char v10; // dl
  unsigned __int8 v11; // al
  __int64 v12; // r14
  unsigned __int8 *j; // r12
  unsigned __int8 **v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r14
  __int64 v17; // rax

  v6 = a1 + 1144;
  v8 = *(unsigned __int8 *)(a1 + 1172);
  do
  {
    if ( !(_BYTE)v8 )
      goto LABEL_8;
    result = *(_QWORD *)(a1 + 1152);
    v8 = *(unsigned int *)(a1 + 1164);
    for ( i = result + 8 * v8; i != result; result += 8 )
    {
      if ( a2 == *(_QWORD *)result )
        return result;
    }
    if ( (unsigned int)v8 < *(_DWORD *)(a1 + 1160) )
    {
      *(_DWORD *)(a1 + 1164) = v8 + 1;
      *(_QWORD *)i = a2;
      v8 = *(unsigned __int8 *)(a1 + 1172);
      ++*(_QWORD *)(a1 + 1144);
    }
    else
    {
LABEL_8:
      result = (__int64)sub_C8CC70(v6, a2, i, v8, a5, a6);
      v8 = *(unsigned __int8 *)(a1 + 1172);
      if ( !v10 )
        return result;
    }
    v11 = *(_BYTE *)(a2 - 16);
    v12 = a2 - 16;
    if ( (v11 & 2) != 0 )
      i = *(_QWORD *)(a2 - 32);
    else
      i = v12 - 8LL * ((v11 >> 2) & 0xF);
    for ( j = *(unsigned __int8 **)i; ; j = (unsigned __int8 *)v17 )
    {
      if ( !(_BYTE)v8 )
        goto LABEL_22;
      v14 = *(unsigned __int8 ***)(a1 + 1152);
      v15 = *(unsigned int *)(a1 + 1164);
      i = (__int64)&v14[v15];
      if ( v14 != (unsigned __int8 **)i )
        break;
LABEL_25:
      if ( (unsigned int)v15 < *(_DWORD *)(a1 + 1160) )
      {
        *(_DWORD *)(a1 + 1164) = v15 + 1;
        *(_QWORD *)i = j;
        v8 = *(unsigned __int8 *)(a1 + 1172);
        ++*(_QWORD *)(a1 + 1144);
        goto LABEL_23;
      }
LABEL_22:
      sub_C8CC70(v6, (__int64)j, i, v8, a5, a6);
      v8 = *(unsigned __int8 *)(a1 + 1172);
      if ( !(_BYTE)i )
        goto LABEL_17;
LABEL_23:
      if ( *j == 18 )
        goto LABEL_17;
      v17 = sub_AF2660(j);
      v8 = *(unsigned __int8 *)(a1 + 1172);
    }
    while ( j != *v14 )
    {
      if ( (unsigned __int8 **)i == ++v14 )
        goto LABEL_25;
    }
LABEL_17:
    result = *(unsigned __int8 *)(a2 - 16);
    if ( (result & 2) != 0 )
    {
      if ( *(_DWORD *)(a2 - 24) == 2 )
      {
        v16 = *(_QWORD *)(a2 - 32);
        goto LABEL_20;
      }
      return result;
    }
    i = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
      break;
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v16 = v12 - result;
LABEL_20:
    a2 = *(_QWORD *)(v16 + 8);
  }
  while ( a2 );
  return result;
}
