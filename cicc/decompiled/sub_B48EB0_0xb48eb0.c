// Function: sub_B48EB0
// Address: 0xb48eb0
//
__int64 __fastcall sub_B48EB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // eax
  __int64 v6; // rax
  __int64 *v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 result; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 66, 0x8000000u, 0, 0);
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = v4 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 72) = v4;
  *(_DWORD *)(a1 + 4) = v5;
  sub_BD2A10(a1, v4, 0);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a1 - 8);
  else
    v6 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v8 )
  {
    v9 = v6 + 32 * v8;
    do
    {
      v10 = *v7;
      if ( *(_QWORD *)v6 )
      {
        v11 = *(_QWORD *)(v6 + 8);
        **(_QWORD **)(v6 + 16) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v6 + 16);
      }
      *(_QWORD *)v6 = v10;
      if ( v10 )
      {
        v12 = *(_QWORD *)(v10 + 16);
        *(_QWORD *)(v6 + 8) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = v6 + 8;
        *(_QWORD *)(v6 + 16) = v10 + 16;
        *(_QWORD *)(v10 + 16) = v6;
      }
      v6 += 32;
      v7 += 4;
    }
    while ( v9 != v6 );
  }
  result = *(_WORD *)(a2 + 2) & 1 | *(_WORD *)(a1 + 2) & 0xFFFEu;
  *(_WORD *)(a1 + 2) = *(_WORD *)(a2 + 2) & 1 | *(_WORD *)(a1 + 2) & 0xFFFE;
  return result;
}
