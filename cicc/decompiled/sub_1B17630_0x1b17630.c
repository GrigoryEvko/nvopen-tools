// Function: sub_1B17630
// Address: 0x1b17630
//
__int64 __fastcall sub_1B17630(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  _QWORD *v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v14; // rdx
  const void *v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  __int64 i; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  __int64 j; // [rsp+30h] [rbp-40h]

  v3 = a2 + 56;
  *(_QWORD *)a1 = a1 + 16;
  v15 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v16 = *(_QWORD *)(v3 - 16);
  for ( i = *(_QWORD *)(v3 - 24); v16 != i; i += 8 )
  {
    v19 = *(_QWORD *)i + 40LL;
    for ( j = *(_QWORD *)(*(_QWORD *)i + 48LL); v19 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      v4 = *(_QWORD *)(j - 16);
      if ( v4 )
      {
        v5 = *(_QWORD **)(a2 + 72);
        do
        {
          v10 = sub_1648700(v4)[5];
          v7 = *(_QWORD **)(a2 + 64);
          if ( v5 == v7 )
          {
            v11 = *(unsigned int *)(a2 + 84);
            v6 = &v5[v11];
            if ( v5 == v6 )
            {
              v14 = v5;
            }
            else
            {
              do
              {
                if ( v10 == *v7 )
                  break;
                ++v7;
              }
              while ( v6 != v7 );
              v14 = &v5[v11];
            }
LABEL_19:
            while ( v14 != v7 )
            {
              if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
                goto LABEL_9;
              ++v7;
            }
            if ( v7 == v6 )
              goto LABEL_21;
          }
          else
          {
            v6 = &v5[*(unsigned int *)(a2 + 80)];
            v7 = sub_16CC9F0(v3, v10);
            if ( v10 == *v7 )
            {
              v5 = *(_QWORD **)(a2 + 72);
              if ( v5 == *(_QWORD **)(a2 + 64) )
                v14 = &v5[*(unsigned int *)(a2 + 84)];
              else
                v14 = &v5[*(unsigned int *)(a2 + 80)];
              goto LABEL_19;
            }
            v5 = *(_QWORD **)(a2 + 72);
            if ( v5 == *(_QWORD **)(a2 + 64) )
            {
              v7 = &v5[*(unsigned int *)(a2 + 84)];
              v14 = v7;
              goto LABEL_19;
            }
            v7 = &v5[*(unsigned int *)(a2 + 80)];
LABEL_9:
            if ( v7 == v6 )
            {
LABEL_21:
              v12 = *(unsigned int *)(a1 + 8);
              if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 12) )
              {
                sub_16CD150(a1, v15, 0, 8, v8, v9);
                v12 = *(unsigned int *)(a1 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) = j - 24;
              ++*(_DWORD *)(a1 + 8);
              break;
            }
          }
          v4 = *(_QWORD *)(v4 + 8);
        }
        while ( v4 );
      }
    }
  }
  return a1;
}
