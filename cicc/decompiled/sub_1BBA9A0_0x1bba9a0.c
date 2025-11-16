// Function: sub_1BBA9A0
// Address: 0x1bba9a0
//
_QWORD *__fastcall sub_1BBA9A0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  _QWORD *result; // rax
  __int64 v7; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 *v11; // r15
  __int64 v12; // r13
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD *v17; // [rsp+18h] [rbp-38h]

  result = a1;
  v7 = (a2 - (__int64)a1) >> 3;
  if ( a2 - (__int64)a1 > 0 )
  {
    while ( 1 )
    {
      v9 = *a3;
      v10 = v7 >> 1;
      v11 = &result[v7 >> 1];
      v12 = *v11;
      if ( *a3 == *v11 || *v11 == 0 || !v9 )
        goto LABEL_17;
      if ( v12 == *(_QWORD *)(v9 + 8) )
        goto LABEL_13;
      if ( v9 == *(_QWORD *)(v12 + 8) || *(_DWORD *)(v12 + 16) >= *(_DWORD *)(v9 + 16) )
        goto LABEL_17;
      v13 = *(_QWORD *)(a4 + 1352);
      if ( *(_BYTE *)(v13 + 72) )
        break;
      v14 = *(_DWORD *)(v13 + 76) + 1;
      *(_DWORD *)(v13 + 76) = v14;
      if ( v14 > 0x20 )
      {
        v16 = a4;
        v17 = result;
        sub_15CC640(v13);
        result = v17;
        a4 = v16;
        if ( *(_DWORD *)(v9 + 48) < *(_DWORD *)(v12 + 48) )
          goto LABEL_17;
LABEL_16:
        if ( *(_DWORD *)(v9 + 52) > *(_DWORD *)(v12 + 52) )
          goto LABEL_17;
LABEL_13:
        result = v11 + 1;
        v7 = v7 - v10 - 1;
        if ( v7 <= 0 )
          return result;
      }
      else
      {
        do
        {
          v15 = v9;
          v9 = *(_QWORD *)(v9 + 8);
        }
        while ( v9 && *(_DWORD *)(v12 + 16) <= *(_DWORD *)(v9 + 16) );
        if ( v12 == v15 )
          goto LABEL_13;
LABEL_17:
        v7 >>= 1;
        if ( v10 <= 0 )
          return result;
      }
    }
    if ( *(_DWORD *)(v9 + 48) < *(_DWORD *)(v12 + 48) )
      goto LABEL_17;
    goto LABEL_16;
  }
  return result;
}
