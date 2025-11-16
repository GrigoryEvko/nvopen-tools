// Function: sub_385DCB0
// Address: 0x385dcb0
//
__int64 __fastcall sub_385DCB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // r14d
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v14; // [rsp+8h] [rbp-48h]
  unsigned int v15; // [rsp+14h] [rbp-3Ch]
  __int64 v16; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v2 = *(_DWORD *)(a2 + 160);
  if ( v2 )
  {
    v15 = 0;
    v14 = 0;
    do
    {
      ++v15;
      v3 = v14;
      v14 = v15;
      if ( v15 >= v2 )
        break;
      v4 = 3 * v3;
      v5 = v15;
      v6 = v15;
      v16 = 16 * v4;
      do
      {
        v7 = *(_QWORD *)(a2 + 152) + v16;
        v8 = *(_QWORD *)(a2 + 152) + 48 * v6;
        if ( sub_385DBF0(a2, v7, v8) )
        {
          v11 = *(unsigned int *)(a1 + 8);
          if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 12) )
          {
            sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, v9, v10);
            v11 = *(unsigned int *)(a1 + 8);
          }
          v12 = (__int64 *)(*(_QWORD *)a1 + 16 * v11);
          *v12 = v7;
          v12[1] = v8;
          ++*(_DWORD *)(a1 + 8);
        }
        v2 = *(_DWORD *)(a2 + 160);
        v6 = v5 + 1;
        v5 = v6;
      }
      while ( (unsigned int)v6 < v2 );
    }
    while ( v15 < v2 );
  }
  return a1;
}
