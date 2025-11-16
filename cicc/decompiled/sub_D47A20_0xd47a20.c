// Function: sub_D47A20
// Address: 0xd47a20
//
unsigned __int64 __fastcall sub_D47A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rbx
  unsigned __int8 *v8; // rdx
  __int64 v9; // r12
  const void *v10; // r8
  __int64 *v11; // rdx
  unsigned __int8 *v12; // rdx
  const void *v13; // [rsp+8h] [rbp-38h]
  const void *v14; // [rsp+8h] [rbp-38h]

  result = **(_QWORD **)(a1 + 32);
  v7 = *(_QWORD *)(result + 16);
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(unsigned __int8 **)(v7 + 24);
      result = (unsigned int)*v8 - 30;
      if ( (unsigned __int8)(*v8 - 30) <= 0xAu )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return result;
    }
    v9 = *((_QWORD *)v8 + 5);
    v10 = (const void *)(a2 + 16);
    if ( !*(_BYTE *)(a1 + 84) )
      goto LABEL_14;
LABEL_4:
    result = *(_QWORD *)(a1 + 64);
    v11 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 76));
    if ( (__int64 *)result != v11 )
    {
      while ( v9 != *(_QWORD *)result )
      {
        result += 8LL;
        if ( v11 == (__int64 *)result )
          goto LABEL_11;
      }
LABEL_8:
      result = *(unsigned int *)(a2 + 8);
      if ( result + 1 > *(unsigned int *)(a2 + 12) )
      {
        v14 = v10;
        sub_C8D5F0(a2, v10, result + 1, 8u, (__int64)v10, a6);
        result = *(unsigned int *)(a2 + 8);
        v10 = v14;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v9;
      ++*(_DWORD *)(a2 + 8);
    }
LABEL_11:
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        break;
      while ( 1 )
      {
        v12 = *(unsigned __int8 **)(v7 + 24);
        result = (unsigned int)*v12 - 30;
        if ( (unsigned __int8)(*v12 - 30) > 0xAu )
          break;
        v9 = *((_QWORD *)v12 + 5);
        if ( *(_BYTE *)(a1 + 84) )
          goto LABEL_4;
LABEL_14:
        v13 = v10;
        result = (unsigned __int64)sub_C8CA60(a1 + 56, v9);
        v10 = v13;
        if ( result )
          goto LABEL_8;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return result;
      }
    }
  }
  return result;
}
