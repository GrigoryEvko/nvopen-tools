// Function: sub_D46D90
// Address: 0xd46d90
//
__int64 *__fastcall sub_D46D90(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  int v6; // r12d
  unsigned int v7; // r15d
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 *v16; // [rsp+28h] [rbp-38h]

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
      v4 = *(_QWORD *)(*v16 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 != *v16 + 48 )
      {
        if ( !v4 )
          BUG();
        v5 = v4 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA )
        {
          v6 = sub_B46E30(v5);
          if ( v6 )
            break;
        }
      }
LABEL_13:
      result = ++v16;
      if ( v14 == v16 )
        return result;
    }
    v7 = 0;
    while ( 1 )
    {
      v8 = sub_B46EC0(v5, v7);
      if ( !*(_BYTE *)(a1 + 84) )
        break;
      v11 = *(_QWORD **)(a1 + 64);
      v12 = &v11[*(unsigned int *)(a1 + 76)];
      if ( v11 == v12 )
        goto LABEL_16;
      while ( v8 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_16;
      }
LABEL_12:
      if ( v6 == ++v7 )
        goto LABEL_13;
    }
    if ( sub_C8CA60(a1 + 56, v8) )
      goto LABEL_12;
LABEL_16:
    v13 = *(unsigned int *)(a2 + 8);
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 8u, v9, v10);
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
