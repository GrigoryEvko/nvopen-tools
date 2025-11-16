// Function: sub_28CF1F0
// Address: 0x28cf1f0
//
__int64 __fastcall sub_28CF1F0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // ecx
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned int v12; // r12d
  __int64 v13; // rax
  __int64 *v14; // r13
  _QWORD **v16; // rdx
  _QWORD *v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx

  v4 = a2;
  if ( *(_BYTE *)a2 == 63 )
    v5 = a2[9];
  else
    v5 = a2[1];
  *(_QWORD *)(a3 + 40) = v5;
  *(_DWORD *)(a3 + 12) = *(unsigned __int8 *)a2 - 29;
  v6 = *(unsigned int *)(a3 + 32);
  if ( *(_DWORD *)(a3 + 32) && (--v6, v6) )
  {
    _BitScanReverse64(&v6, v6);
    v7 = 64 - (v6 ^ 0x3F);
    v6 = (int)v7;
    if ( *(_DWORD *)(a1 + 176) <= v7 )
      goto LABEL_6;
  }
  else
  {
    LOBYTE(v7) = 0;
    if ( !*(_DWORD *)(a1 + 176) )
      goto LABEL_6;
  }
  v16 = (_QWORD **)(*(_QWORD *)(a1 + 168) + 8 * v6);
  v17 = *v16;
  if ( *v16 )
  {
    *v16 = (_QWORD *)*v17;
    goto LABEL_14;
  }
LABEL_6:
  v8 = *(_QWORD *)(a1 + 72);
  v9 = 8LL << v7;
  *(_QWORD *)(a1 + 152) += 8LL << v7;
  v10 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (8LL << v7) + v10;
  if ( *(_QWORD *)(a1 + 80) >= v11 && v8 )
  {
    *(_QWORD *)(a1 + 72) = v11;
    *(_QWORD *)(a3 + 24) = v10;
    if ( (*((_BYTE *)v4 + 7) & 0x40) == 0 )
      goto LABEL_9;
    goto LABEL_15;
  }
  v17 = (_QWORD *)sub_9D1E70(a1 + 72, v9, v9, 3);
LABEL_14:
  *(_QWORD *)(a3 + 24) = v17;
  if ( (*((_BYTE *)v4 + 7) & 0x40) == 0 )
  {
LABEL_9:
    v12 = 1;
    v13 = 32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF);
    v14 = &v4[v13 / 0xFFFFFFFFFFFFFFF8LL];
    if ( &v4[v13 / 0xFFFFFFFFFFFFFFF8LL] == v4 )
      return v12;
    goto LABEL_16;
  }
LABEL_15:
  v14 = (__int64 *)*(v4 - 1);
  v12 = 1;
  v4 = &v14[4 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  if ( v14 == v4 )
    return v12;
  do
  {
LABEL_16:
    v18 = (_BYTE *)sub_28C86C0(a1, *v14);
    v19 = v18;
    if ( (_BYTE)v12 )
      LOBYTE(v12) = *v18 <= 0x15u;
    v20 = *(unsigned int *)(a3 + 36);
    v21 = *(_QWORD *)(a3 + 24);
    v14 += 4;
    *(_DWORD *)(a3 + 36) = v20 + 1;
    *(_QWORD *)(v21 + 8 * v20) = v19;
  }
  while ( v14 != v4 );
  return v12;
}
