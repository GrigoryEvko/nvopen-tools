// Function: sub_D46840
// Address: 0xd46840
//
__int64 __fastcall sub_D46840(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 56);
  v17 = a2 + 48;
  if ( v6 == a2 + 48 )
    return 1;
  while ( 1 )
  {
    v7 = v6 - 24;
    if ( !v6 )
      v7 = 0;
    if ( !a4 || *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) != 11 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      if ( v8 )
        break;
    }
LABEL_16:
    v6 = *(_QWORD *)(v6 + 8);
    if ( v17 == v6 )
      return 1;
  }
  while ( 1 )
  {
    v9 = *(_QWORD *)(v8 + 24);
    v10 = *(_QWORD *)(v9 + 40);
    if ( *(_BYTE *)v9 == 84 )
      v10 = *(_QWORD *)(*(_QWORD *)(v9 - 8)
                      + 32LL * *(unsigned int *)(v9 + 72)
                      + 8LL * (unsigned int)((v8 - *(_QWORD *)(v9 - 8)) >> 5));
    if ( v10 == a2 )
      goto LABEL_15;
    if ( *(_BYTE *)(a1 + 84) )
      break;
    v19 = v8;
    v14 = sub_C8CA60(a1 + 56, v10);
    v8 = v19;
    if ( !v14 )
      goto LABEL_19;
LABEL_15:
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      goto LABEL_16;
  }
  v11 = *(_QWORD **)(a1 + 64);
  v12 = &v11[*(unsigned int *)(a1 + 76)];
  if ( v11 != v12 )
  {
    while ( v10 != *v11 )
    {
      if ( v12 == ++v11 )
        goto LABEL_19;
    }
    goto LABEL_15;
  }
LABEL_19:
  v15 = 0;
  v16 = 0;
  if ( v10 )
  {
    v15 = (unsigned int)(*(_DWORD *)(v10 + 44) + 1);
    v16 = *(_DWORD *)(v10 + 44) + 1;
  }
  if ( v16 >= *(_DWORD *)(a3 + 32) || !*(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v15) )
    goto LABEL_15;
  return 0;
}
