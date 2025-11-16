// Function: sub_19E5840
// Address: 0x19e5840
//
__int64 __fastcall sub_19E5840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 ****v4; // rbx
  __int64 ***v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  char v8; // cl
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 ****v11; // r13
  _QWORD **v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx

  v4 = (__int64 ****)a2;
  if ( *(_BYTE *)(a2 + 16) == 56 )
    v5 = *(__int64 ****)(a2 + 56);
  else
    v5 = *(__int64 ****)a2;
  *(_QWORD *)(a3 + 40) = v5;
  *(_DWORD *)(a3 + 12) = *(unsigned __int8 *)(a2 + 16) - 24;
  v6 = *(unsigned int *)(a3 + 32);
  if ( *(_DWORD *)(a3 + 32) )
  {
    if ( --v6 )
    {
      _BitScanReverse64(&v6, v6);
      v7 = 64 - (v6 ^ 0x3F);
      v8 = 64 - (v6 ^ 0x3F);
      v6 = v7;
      if ( v7 >= *(_DWORD *)(a1 + 176) )
        goto LABEL_6;
      goto LABEL_11;
    }
    v8 = 0;
  }
  else
  {
    v8 = 0;
  }
  if ( !*(_DWORD *)(a1 + 176) )
    goto LABEL_6;
LABEL_11:
  v13 = (_QWORD **)(*(_QWORD *)(a1 + 168) + 8 * v6);
  v14 = *v13;
  if ( !*v13 )
  {
LABEL_6:
    *(_QWORD *)(a3 + 24) = sub_145CBF0((__int64 *)(a1 + 64), 8LL << v8, 8);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
      goto LABEL_7;
    goto LABEL_13;
  }
  *v13 = (_QWORD *)*v14;
  *(_QWORD *)(a3 + 24) = v14;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
  {
LABEL_7:
    v9 = 1;
    v10 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v11 = (__int64 ****)(a2 - v10);
    if ( a2 - v10 == a2 )
      return v9;
    goto LABEL_14;
  }
LABEL_13:
  v11 = *(__int64 *****)(a2 - 8);
  v9 = 1;
  v4 = &v11[3 * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)];
  if ( v11 == v4 )
    return v9;
  do
  {
LABEL_14:
    v15 = sub_19E1ED0(a1, *v11);
    v16 = v15;
    if ( (_BYTE)v9 )
      LOBYTE(v9) = *(_BYTE *)(v15 + 16) <= 0x10u;
    v17 = *(unsigned int *)(a3 + 36);
    v18 = *(_QWORD *)(a3 + 24);
    v11 += 3;
    *(_DWORD *)(a3 + 36) = v17 + 1;
    *(_QWORD *)(v18 + 8 * v17) = v16;
  }
  while ( v11 != v4 );
  return v9;
}
