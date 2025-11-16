// Function: sub_B48BF0
// Address: 0xb48bf0
//
__int64 __fastcall sub_B48BF0(__int64 a1, unsigned int a2, char a3)
{
  __int64 v3; // r9
  unsigned int v4; // r10d
  __int64 v5; // rsi
  int v8; // edx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // r13
  __int64 v14; // r8
  __int64 v15; // rsi
  __int64 i; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  const void *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // eax
  int v25; // edx
  __int64 v27; // rax

  v3 = a2;
  v4 = a2 + 1;
  v5 = 32LL * a2;
  v8 = *(_DWORD *)(a1 + 4);
  v9 = *(_QWORD *)(a1 - 8);
  v10 = v8 & 0x7FFFFFF;
  v11 = (__int64 *)(v9 + v5);
  v5 += 32;
  v12 = (__int64 *)(v9 + v5);
  v13 = *v11;
  v14 = 32 * v10 - v5;
  v15 = v14 >> 5;
  if ( v14 > 0 )
  {
    for ( i = *v11; ; i = *v11 )
    {
      v17 = *v12;
      if ( i )
      {
        v18 = v11[1];
        *(_QWORD *)v11[2] = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = v11[2];
      }
      *v11 = v17;
      if ( v17 )
      {
        v19 = *(_QWORD *)(v17 + 16);
        v11[1] = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = v11 + 1;
        v11[2] = v17 + 16;
        *(_QWORD *)(v17 + 16) = v11;
      }
      v12 += 4;
      v11 += 4;
      if ( !--v15 )
        break;
    }
    v9 = *(_QWORD *)(a1 - 8);
    v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  v20 = 32LL * *(unsigned int *)(a1 + 72);
  v21 = (const void *)(v9 + v20 + 8LL * v4);
  if ( v21 != (const void *)(v9 + v20 + 8 * v10) )
  {
    memmove((void *)(v20 + 8 * v3 + v9), v21, 8 * (v10 - v4));
    v9 = *(_QWORD *)(a1 - 8);
    v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  v22 = v9 + 32 * v10 - 32;
  if ( *(_QWORD *)v22 )
  {
    v23 = *(_QWORD *)(v22 + 8);
    **(_QWORD **)(v22 + 16) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
  }
  *(_QWORD *)v22 = 0;
  v24 = *(_DWORD *)(a1 + 4);
  v25 = (v24 + 0x7FFFFFF) & 0x7FFFFFF;
  *(_DWORD *)(a1 + 4) = v25 | v24 & 0xF8000000;
  if ( v25 || !a3 )
    return v13;
  v27 = sub_ACADE0(*(__int64 ***)(a1 + 8));
  sub_BD84D0(a1, v27);
  sub_B43D60((_QWORD *)a1);
  return v13;
}
