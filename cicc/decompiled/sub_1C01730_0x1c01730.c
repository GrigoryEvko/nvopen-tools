// Function: sub_1C01730
// Address: 0x1c01730
//
__int64 __fastcall sub_1C01730(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v5; // r13d
  __int64 v6; // r12
  unsigned int v7; // eax
  int v8; // r8d
  int v9; // r9d
  int v10; // ecx
  __int64 v11; // rdx
  __int64 v13; // rax
  unsigned int v14; // [rsp+4h] [rbp-3Ch]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v15 = a2 + 72;
  if ( v3 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v6 = 0;
        if ( v3 )
          v6 = v3 - 24;
        v7 = sub_1C014D0(a1, v6, 0);
        if ( v7 <= v5 )
          break;
        v10 = *(_DWORD *)(a3 + 12);
        *(_DWORD *)(a3 + 8) = 0;
        v11 = 0;
        if ( !v10 )
        {
          v14 = v7;
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v8, v9);
          v7 = v14;
          v11 = 8LL * *(unsigned int *)(a3 + 8);
        }
        v5 = v7;
        *(_QWORD *)(*(_QWORD *)a3 + v11) = v6;
        ++*(_DWORD *)(a3 + 8);
        v3 = *(_QWORD *)(v3 + 8);
        if ( v15 == v3 )
          return v5;
      }
      if ( v7 == v5 )
      {
        v13 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v8, v9);
          v13 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v6;
        ++*(_DWORD *)(a3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v15 != v3 );
  }
  return v5;
}
