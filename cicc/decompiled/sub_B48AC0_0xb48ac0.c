// Function: sub_B48AC0
// Address: 0xb48ac0
//
__int64 __fastcall sub_B48AC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // eax
  __int64 v6; // rdi
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // r8
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 result; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 55, 0x8000000u, 0, 0);
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = v4 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 72) = v4;
  *(_DWORD *)(a1 + 4) = v5;
  sub_BD2A10(a1, v4, 1);
  v6 = *(_QWORD *)(a1 - 8);
  v7 = *(__int64 **)(a2 - 8);
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( 32 * v8 )
  {
    v9 = &v7[4 * v8];
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
      v7 += 4;
      v6 += 32;
    }
    while ( v9 != v7 );
    v6 = *(_QWORD *)(a1 - 8);
    v7 = *(__int64 **)(a2 - 8);
    v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  v13 = 4LL * *(unsigned int *)(a2 + 72);
  if ( &v7[v13] != &v7[v8 + v13] )
    memmove((void *)(32LL * *(unsigned int *)(a1 + 72) + v6), &v7[v13], 8 * v8);
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
