// Function: sub_24F3620
// Address: 0x24f3620
//
__int64 __fastcall sub_24F3620(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rax
  __int64 **v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  __int64 v10; // rax
  _QWORD **v11; // rbx
  _QWORD **v12; // r14
  _QWORD *v13; // r15
  __int64 *v14; // r12
  __int64 *i; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 *v20; // rbx
  __int64 result; // rax
  __int64 *j; // r12
  __int64 v23; // rdi
  __int64 v24; // rax

  v5 = (__int64 *)sub_B2BE50(a2);
  v6 = (__int64 **)sub_BCE3C0(v5, 0);
  v9 = sub_ACADE0(v6);
  v10 = a3;
  v11 = *(_QWORD ***)a3;
  v12 = &v11[*(unsigned int *)(v10 + 8)];
  while ( v12 != v11 )
  {
    v13 = *v11++;
    sub_BD84D0((__int64)v13, v9);
    sub_B43D60(v13);
  }
  *(_DWORD *)(a3 + 8) = 0;
  v14 = *(__int64 **)(a1 + 120);
  for ( i = &v14[*(unsigned int *)(a1 + 128)]; i != v14; ++v14 )
  {
    v16 = *v14;
    v17 = sub_ACADE0(*(__int64 ***)(*v14 + 8));
    sub_BD84D0(v16, v17);
    sub_B43D60((_QWORD *)v16);
    v18 = *(_QWORD *)(v16 - 32);
    if ( !v18 || *(_BYTE *)v18 || *(_QWORD *)(v18 + 24) != *(_QWORD *)(v16 + 80) )
      BUG();
    if ( *(_DWORD *)(v18 + 36) == 60 )
    {
      v19 = *(_QWORD **)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v19 == 85 )
      {
        v24 = *(v19 - 4);
        if ( v24 )
        {
          if ( !*(_BYTE *)v24
            && *(_QWORD *)(v24 + 24) == v19[10]
            && (*(_BYTE *)(v24 + 33) & 0x20) != 0
            && *(_DWORD *)(v24 + 36) == 57 )
          {
            sub_B43D60(v19);
          }
        }
      }
    }
  }
  v20 = *(__int64 **)(a1 + 8);
  result = *(unsigned int *)(a1 + 16);
  *(_DWORD *)(a1 + 128) = 0;
  for ( j = &v20[result]; j != v20; result = sub_F55BE0(v23, 0, 0, 0, v7, v8) )
    v23 = *v20++;
  return result;
}
