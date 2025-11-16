// Function: sub_15F9CE0
// Address: 0x15f9ce0
//
__int64 __fastcall sub_15F9CE0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  _QWORD *v8; // rax
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r9
  unsigned __int64 v16; // rdi
  __int64 v17; // rdi

  v8 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *v8 )
  {
    v9 = v8[1];
    v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *v8 = a2;
  if ( a2 )
  {
    v11 = *(_QWORD *)(a2 + 8);
    v8[1] = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
    v8[2] = (a2 + 8) | v8[2] & 3LL;
    *(_QWORD *)(a2 + 8) = v8;
  }
  v12 = (_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  v13 = (8 * a4) >> 3;
  if ( 8 * a4 > 0 )
  {
    do
    {
      v14 = *a3;
      if ( *v12 )
      {
        v15 = v12[1];
        v16 = v12[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v16 = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
      }
      *v12 = v14;
      if ( v14 )
      {
        v17 = *(_QWORD *)(v14 + 8);
        v12[1] = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = (unsigned __int64)(v12 + 1) | *(_QWORD *)(v17 + 16) & 3LL;
        v12[2] = (v14 + 8) | v12[2] & 3LL;
        *(_QWORD *)(v14 + 8) = v12;
      }
      ++a3;
      v12 += 3;
      --v13;
    }
    while ( v13 );
  }
  return sub_164B780(a1, a5);
}
