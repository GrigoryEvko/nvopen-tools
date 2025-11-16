// Function: sub_D36170
// Address: 0xd36170
//
__int64 __fastcall sub_D36170(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // r13d
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // r9
  char v15; // al
  __int64 *v17; // rax
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned int v19; // [rsp+14h] [rbp-3Ch]
  __int64 v20; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v2 = *(_DWORD *)(a2 + 176);
  if ( v2 )
  {
    v19 = 0;
    v18 = 0;
    do
    {
      ++v19;
      v3 = v18;
      v18 = v19;
      if ( v19 >= v2 )
        break;
      v4 = 3 * v3;
      v5 = v19;
      v6 = v19;
      v20 = 16 * v4;
      do
      {
        v11 = *(_QWORD *)(a2 + 168) + v20;
        v12 = *(_QWORD *)(a2 + 168) + 48 * v6;
        if ( sub_D34690(a2, v11, v12) )
        {
          v15 = *(_BYTE *)(a2 + 376);
          if ( v15 )
            v15 = sub_D36150(a2, v11, v12);
          *(_BYTE *)(a2 + 376) = v15;
          v7 = *(unsigned int *)(a1 + 8);
          v8 = *(unsigned int *)(a1 + 12);
          v9 = *(_DWORD *)(a1 + 8);
          if ( v7 >= v8 )
          {
            if ( v8 < v7 + 1 )
            {
              sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + 1, 0x10u, v13, v14);
              v7 = *(unsigned int *)(a1 + 8);
            }
            v17 = (__int64 *)(*(_QWORD *)a1 + 16 * v7);
            *v17 = v11;
            v17[1] = v12;
            ++*(_DWORD *)(a1 + 8);
          }
          else
          {
            v10 = (__int64 *)(*(_QWORD *)a1 + 16 * v7);
            if ( v10 )
            {
              *v10 = v11;
              v10[1] = v12;
              v9 = *(_DWORD *)(a1 + 8);
            }
            *(_DWORD *)(a1 + 8) = v9 + 1;
          }
        }
        v2 = *(_DWORD *)(a2 + 176);
        v6 = v5 + 1;
        v5 = v6;
      }
      while ( v2 > (unsigned int)v6 );
    }
    while ( v2 > v19 );
  }
  return a1;
}
