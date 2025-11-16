// Function: sub_B54710
// Address: 0xb54710
//
__int64 __fastcall sub_B54710(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 result; // rax

  v4 = sub_BD5C60(a2, a2);
  v5 = sub_BCB120(v4);
  sub_B44260(a1, v5, 4, 0x8000000u, 0, 0);
  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  *(_DWORD *)(a1 + 4) = v6 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  sub_BD2A10(a1, v6, 0);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a1 - 8);
  else
    v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v9 = *(__int64 **)(a2 - 8);
  else
    v9 = (__int64 *)(a2 - 32LL * (unsigned int)v8);
  if ( (_DWORD)v8 )
  {
    v10 = v7 + 32 * v8;
    do
    {
      v11 = *v9;
      if ( *(_QWORD *)v7 )
      {
        v12 = *(_QWORD *)(v7 + 8);
        **(_QWORD **)(v7 + 16) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v7 + 16);
      }
      *(_QWORD *)v7 = v11;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v7 + 8) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = v7 + 8;
        *(_QWORD *)(v7 + 16) = v11 + 16;
        *(_QWORD *)(v11 + 16) = v7;
      }
      v7 += 32;
      v9 += 4;
    }
    while ( v7 != v10 );
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
