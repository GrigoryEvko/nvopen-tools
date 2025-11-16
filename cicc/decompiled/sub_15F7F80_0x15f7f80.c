// Function: sub_15F7F80
// Address: 0x15f7f80
//
__int64 __fastcall sub_15F7F80(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r9
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax

  v9 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v10 = (8 * a4) >> 3;
  if ( 8 * a4 > 0 )
  {
    do
    {
      v11 = *a3;
      if ( *v9 )
      {
        v12 = v9[1];
        v13 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *v9 = v11;
      if ( v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        v9[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (unsigned __int64)(v9 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
        v9[2] = (v11 + 8) | v9[2] & 3LL;
        *(_QWORD *)(v11 + 8) = v9;
      }
      ++a3;
      v9 += 3;
      --v10;
    }
    while ( v10 );
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v15 = *(_QWORD *)(a1 - 16);
    v16 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v16 = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
  }
  *(_QWORD *)(a1 - 24) = a2;
  if ( a2 )
  {
    v17 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = (a1 - 16) | *(_QWORD *)(v17 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a2 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a2 + 8) = a1 - 24;
  }
  return sub_164B780(a1, a5);
}
