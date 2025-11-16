// Function: sub_B4BAA0
// Address: 0xb4baa0
//
__int64 __fastcall sub_B4BAA0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx

  v4 = sub_BD5C60(a2, a2);
  v5 = sub_BCB120(v4);
  sub_B44260(a1, v5, 1, a3, 0, 0);
  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v6 )
  {
    v8 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v9 = *(_QWORD *)(a2 - 32LL * v6);
    if ( *(_QWORD *)v8 )
    {
      v10 = *(_QWORD *)(v8 + 8);
      **(_QWORD **)(v8 + 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v8 + 16);
    }
    *(_QWORD *)v8 = v9;
    if ( v9 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v8 + 8) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v8 + 8;
      *(_QWORD *)(v8 + 16) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v8;
    }
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
