// Function: sub_B4DAA0
// Address: 0xb4daa0
//
__int64 __fastcall sub_B4DAA0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // edi
  int v6; // edx
  __int64 v7; // rdi
  __int64 *v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 result; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 34, a3, 0, 0);
  v5 = *(_DWORD *)(a2 + 4);
  v6 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
  v7 = 32LL * (v5 & 0x7FFFFFF);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  v8 = (__int64 *)(a2 - v7);
  v9 = a1 - 32LL * (v6 & 0x7FFFFFF);
  if ( v7 )
  {
    v10 = v9 + v7;
    do
    {
      v11 = *v8;
      if ( *(_QWORD *)v9 )
      {
        v12 = *(_QWORD *)(v9 + 8);
        **(_QWORD **)(v9 + 16) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v9 + 16);
      }
      *(_QWORD *)v9 = v11;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v9 + 8) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = v9 + 8;
        *(_QWORD *)(v9 + 16) = v11 + 16;
        *(_QWORD *)(v11 + 16) = v9;
      }
      v9 += 32;
      v8 += 4;
    }
    while ( v10 != v9 );
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
