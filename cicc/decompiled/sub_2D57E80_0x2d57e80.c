// Function: sub_2D57E80
// Address: 0x2d57e80
//
void __fastcall sub_2D57E80(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r8
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = 8 * v1;
    v3 = 0;
    do
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + v3);
      if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
        v4 = *(_QWORD *)(v8 - 8);
      else
        v4 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
      v5 = v4 + 4 * v3;
      if ( *(_QWORD *)v5 )
      {
        v6 = *(_QWORD *)(v5 + 8);
        **(_QWORD **)(v5 + 16) = v6;
        if ( v6 )
          *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 + 16);
      }
      *(_QWORD *)v5 = v9;
      if ( v9 )
      {
        v7 = *(_QWORD *)(v9 + 16);
        *(_QWORD *)(v5 + 8) = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = v5 + 8;
        *(_QWORD *)(v5 + 16) = v9 + 16;
        *(_QWORD *)(v9 + 16) = v5;
      }
      v3 += 8;
    }
    while ( v2 != v3 );
  }
}
