// Function: sub_2C03DE0
// Address: 0x2c03de0
//
__int64 *__fastcall sub_2C03DE0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 **v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r13
  signed __int64 v11; // r12
  int v12; // r12d
  __int64 v13; // rax

  result = a2;
  v7 = (__int64 **)(a1 + 2);
  *a1 = a1 + 2;
  a1[1] = 0x800000000LL;
  v8 = result[3];
  v9 = result[1];
  v10 = *result;
  if ( v9 == v8 && result[2] == *result )
  {
    v12 = 0;
  }
  else
  {
    v11 = 0;
    do
    {
      do
        ++v11;
      while ( v9 - v11 != v8 );
    }
    while ( result[2] != *result );
    if ( v11 > 8 )
    {
      sub_C8D5F0((__int64)a1, v7, v11, 8u, a5, a6);
      v7 = (__int64 **)(*a1 + 8LL * *((unsigned int *)a1 + 2));
    }
    do
    {
      --v9;
      if ( *(_BYTE *)(v10 + 8) )
      {
        v13 = v10;
        while ( !*(_DWORD *)(v13 + 88) )
        {
          v13 = *(_QWORD *)(v13 + 48);
          if ( !v13 )
            BUG();
        }
        result = *(__int64 **)(*(_QWORD *)(v13 + 80) + 8LL * (unsigned int)v9);
      }
      else
      {
        result = *(__int64 **)(v10 + 112);
      }
      if ( v7 )
        *v7 = result;
      ++v7;
    }
    while ( v8 != v9 );
    v12 = *((_DWORD *)a1 + 2) + v11;
  }
  *((_DWORD *)a1 + 2) = v12;
  return result;
}
