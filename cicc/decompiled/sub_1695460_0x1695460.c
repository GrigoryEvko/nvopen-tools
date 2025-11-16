// Function: sub_1695460
// Address: 0x1695460
//
__int64 __fastcall sub_1695460(__int64 a1, int a2, unsigned int a3, __int64 a4, _DWORD *a5, _QWORD *a6)
{
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rdi
  _WORD *v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rax
  __int64 v27; // r9
  bool v28; // cc
  _QWORD *v29; // rsi
  __int64 v30; // r10
  __int64 v31; // r9
  __int64 v32; // rsi
  __int64 v33; // rsi
  unsigned int v34; // [rsp+4h] [rbp-3Ch]
  __int64 v35; // [rsp+8h] [rbp-38h]

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v10 = sub_1625790(a1, 2);
  if ( !v10 )
    return 0;
  v11 = *(_DWORD *)(v10 + 8);
  if ( v11 <= 4 )
    return 0;
  v34 = *(_DWORD *)(v10 + 8);
  v35 = v10;
  v12 = *(_QWORD *)(v10 - 8LL * v11);
  if ( !v12 )
    return 0;
  v13 = (_WORD *)sub_161E970(v12);
  if ( v14 != 2 )
    return 0;
  if ( *v13 != 20566 )
    return 0;
  v16 = *(unsigned int *)(v35 + 8);
  v17 = *(_QWORD *)(v35 + 8 * (1 - v16));
  if ( *(_BYTE *)v17 != 1 )
    return 0;
  v18 = *(_QWORD *)(v17 + 136);
  if ( *(_BYTE *)(v18 + 16) != 13 )
    return 0;
  v19 = *(_DWORD *)(v18 + 32) <= 0x40u ? *(_QWORD *)(v18 + 24) : **(_QWORD **)(v18 + 24);
  if ( a2 != v19 )
    return 0;
  v20 = *(_QWORD *)(v35 + 8 * (2 - v16));
  if ( *(_BYTE *)v20 != 1 )
    return 0;
  v21 = *(_QWORD *)(v20 + 136);
  if ( *(_BYTE *)(v21 + 16) != 13 )
    return 0;
  if ( *(_DWORD *)(v21 + 32) <= 0x40u )
    v22 = *(_QWORD *)(v21 + 24);
  else
    v22 = **(_QWORD **)(v21 + 24);
  *a6 = v22;
  v23 = 3;
  v24 = 0;
  *a5 = 0;
  do
  {
    if ( a3 <= (unsigned int)v24 )
      break;
    v30 = *(unsigned int *)(v35 + 8);
    v31 = 0;
    v32 = *(_QWORD *)(v35 + 8 * (v23 - v30));
    if ( *(_BYTE *)v32 == 1 )
    {
      v31 = *(_QWORD *)(v32 + 136);
      if ( *(_BYTE *)(v31 + 16) != 13 )
        v31 = 0;
    }
    v33 = *(_QWORD *)(v35 + 8 * ((unsigned int)(v23 + 1) - v30));
    if ( *(_BYTE *)v33 != 1 )
      return 0;
    v25 = *(_QWORD *)(v33 + 136);
    if ( *(_BYTE *)(v25 + 16) != 13 || !v31 )
      return 0;
    v26 = (_QWORD *)(a4 + 16 * v24);
    v27 = *(_DWORD *)(v31 + 32) <= 0x40u ? *(_QWORD *)(v31 + 24) : **(_QWORD **)(v31 + 24);
    *v26 = v27;
    v28 = *(_DWORD *)(v25 + 32) <= 0x40u;
    v29 = *(_QWORD **)(v25 + 24);
    if ( !v28 )
      v29 = (_QWORD *)*v29;
    *(_QWORD *)(a4 + 16LL * (unsigned int)*a5 + 8) = v29;
    v23 += 2;
    v24 = (unsigned int)(*a5 + 1);
    *a5 = v24;
  }
  while ( v34 > (unsigned int)v23 );
  return 1;
}
