// Function: sub_2EA42C0
// Address: 0x2ea42c0
//
__int64 *__fastcall sub_2EA42C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 *v16; // [rsp+18h] [rbp-38h]

  result = *(__int64 **)(a1 + 40);
  v14 = result;
  v16 = *(__int64 **)(a1 + 32);
  if ( result == v16 )
    return result;
  do
  {
    while ( 1 )
    {
      v15 = *v16;
      v8 = *(__int64 **)(*v16 + 112);
      v9 = &v8[*(unsigned int *)(*v16 + 120)];
      if ( v8 != v9 )
        break;
LABEL_9:
      result = ++v16;
      if ( v14 == v16 )
        return result;
    }
    while ( 1 )
    {
      v10 = *v8;
      if ( !*(_BYTE *)(a1 + 84) )
        break;
      v11 = *(_QWORD **)(a1 + 64);
      v12 = &v11[*(unsigned int *)(a1 + 76)];
      if ( v11 == v12 )
        goto LABEL_12;
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_12;
      }
LABEL_8:
      if ( v9 == ++v8 )
        goto LABEL_9;
    }
    if ( sub_C8CA60(a1 + 56, v10) )
      goto LABEL_8;
LABEL_12:
    v13 = *(unsigned int *)(a2 + 8);
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 8u, a5, a6);
      v13 = *(unsigned int *)(a2 + 8);
    }
    ++v16;
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v15;
    result = v16;
    ++*(_DWORD *)(a2 + 8);
  }
  while ( v14 != v16 );
  return result;
}
