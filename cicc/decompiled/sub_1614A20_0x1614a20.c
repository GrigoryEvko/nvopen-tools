// Function: sub_1614A20
// Address: 0x1614a20
//
__int64 __fastcall sub_1614A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r15
  __int64 *i; // r13
  __int64 v7; // r10
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 *v14; // rdx
  __int64 result; // rax
  __int64 *v16; // r13
  __int64 *v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r10
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+18h] [rbp-38h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v23 = sub_16135E0(*(_QWORD *)(a1 + 16), a4);
  v5 = *(__int64 **)(v23 + 144);
  for ( i = &v5[*(unsigned int *)(v23 + 152)]; i != v5; ++v5 )
  {
    v7 = sub_160E9B0(a1, *v5, 1);
    if ( v7 )
    {
      v8 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 12) )
      {
        v21 = v7;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v8 = *(unsigned int *)(a2 + 8);
        v7 = v21;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v8) = v7;
      ++*(_DWORD *)(a2 + 8);
    }
  }
  v9 = *(_QWORD *)v23 + 8LL * *(unsigned int *)(v23 + 8);
  if ( v9 != *(_QWORD *)v23 )
  {
    v10 = *(__int64 **)v23;
    do
    {
      while ( 1 )
      {
        v20 = sub_160E9B0(a1, *v10, 1);
        if ( v20 )
          break;
        v12 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, a3 + 16, 0, 8);
          v12 = *(unsigned int *)(a3 + 8);
        }
        v13 = *v10++;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
        ++*(_DWORD *)(a3 + 8);
        if ( (__int64 *)v9 == v10 )
          goto LABEL_16;
      }
      v11 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 12) )
      {
        v22 = v20;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v11 = *(unsigned int *)(a2 + 8);
        v20 = v22;
      }
      ++v10;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v20;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( (__int64 *)v9 != v10 );
  }
LABEL_16:
  v14 = *(__int64 **)(v23 + 80);
  result = *(unsigned int *)(v23 + 88);
  v16 = &v14[result];
  if ( v16 != v14 )
  {
    v17 = *(__int64 **)(v23 + 80);
    do
    {
      while ( 1 )
      {
        result = sub_160E9B0(a1, *v17, 1);
        if ( result )
          break;
        result = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, a3 + 16, 0, 8);
          result = *(unsigned int *)(a3 + 8);
        }
        v19 = *v17++;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v19;
        ++*(_DWORD *)(a3 + 8);
        if ( v16 == v17 )
          return result;
      }
      v18 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v18 >= *(_DWORD *)(a2 + 12) )
      {
        v24 = result;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v18 = *(unsigned int *)(a2 + 8);
        result = v24;
      }
      ++v17;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v18) = result;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v16 != v17 );
  }
  return result;
}
