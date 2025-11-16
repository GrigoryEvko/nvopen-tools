// Function: sub_14A88D0
// Address: 0x14a88d0
//
void __fastcall sub_14A88D0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 i; // r15
  __int64 v8; // r15
  __int64 *v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+8h] [rbp-48h]

  v6 = *(_QWORD *)(a5 + 24 * (1LL - (*(_DWORD *)(a5 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v6 + 16) != 13 )
  {
    *a4 = 1;
    return;
  }
  for ( i = *(_QWORD *)(a5 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v12 = sub_1648700(i);
    if ( *(_BYTE *)(v12 + 16) != 86 || *(_DWORD *)(v12 + 64) != 1 )
      goto LABEL_9;
    v13 = **(_DWORD **)(v12 + 56);
    if ( !v13 )
    {
      v15 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 12) )
      {
        v16 = v12;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v15 = *(unsigned int *)(a2 + 8);
        v12 = v16;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v12;
      ++*(_DWORD *)(a2 + 8);
      continue;
    }
    if ( v13 == 1 )
    {
      v14 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v14 >= *(_DWORD *)(a3 + 12) )
      {
        v17 = v12;
        sub_16CD150(a3, a3 + 16, 0, 8);
        v14 = *(unsigned int *)(a3 + 8);
        v12 = v17;
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v12;
      ++*(_DWORD *)(a3 + 8);
    }
    else
    {
LABEL_9:
      *a4 = 1;
    }
  }
  v8 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v9 = *(__int64 **)a2;
  if ( *(_QWORD *)a2 != v8 )
  {
    do
    {
      v10 = *v9;
      v11 = *(_QWORD **)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
        v11 = (_QWORD *)*v11;
      ++v9;
      sub_14A8490(a1, a4, *(_QWORD *)(v10 + 8), (__int64)v11);
    }
    while ( (__int64 *)v8 != v9 );
  }
}
