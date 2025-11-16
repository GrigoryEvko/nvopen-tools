// Function: sub_B88D90
// Address: 0xb88d90
//
__int64 __fastcall sub_B88D90(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v5; // r15
  __int64 *i; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // r13
  __int64 *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 v15; // [rsp+18h] [rbp-38h]
  __int64 v16; // [rsp+18h] [rbp-38h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  v15 = sub_B873F0(*(_QWORD *)(a1 + 8), a4);
  v5 = *(__int64 **)(v15 + 144);
  for ( i = &v5[*(unsigned int *)(v15 + 152)]; i != v5; ++v5 )
  {
    v7 = sub_B81110(a1, *v5, 1);
    if ( v7 )
    {
      v8 = *(unsigned int *)(a2 + 8);
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v14 = v7;
        sub_C8D5F0(a2, a2 + 16, v8 + 1, 8);
        v8 = *(unsigned int *)(a2 + 8);
        v7 = v14;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = v7;
      ++*(_DWORD *)(a2 + 8);
    }
  }
  result = *(unsigned int *)(v15 + 8);
  v10 = *(_QWORD *)v15 + 8 * result;
  if ( v10 != *(_QWORD *)v15 )
  {
    v11 = *(__int64 **)v15;
    do
    {
      while ( 1 )
      {
        result = sub_B81110(a1, *v11, 1);
        if ( result )
          break;
        result = *(unsigned int *)(a3 + 8);
        v13 = *v11;
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v17 = *v11;
          sub_C8D5F0(a3, a3 + 16, result + 1, 8);
          result = *(unsigned int *)(a3 + 8);
          v13 = v17;
        }
        ++v11;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v13;
        ++*(_DWORD *)(a3 + 8);
        if ( (__int64 *)v10 == v11 )
          return result;
      }
      v12 = *(unsigned int *)(a2 + 8);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v16 = result;
        sub_C8D5F0(a2, a2 + 16, v12 + 1, 8);
        v12 = *(unsigned int *)(a2 + 8);
        result = v16;
      }
      ++v11;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = result;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( (__int64 *)v10 != v11 );
  }
  return result;
}
