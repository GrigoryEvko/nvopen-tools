// Function: sub_1D5B170
// Address: 0x1d5b170
//
void __fastcall sub_1D5B170(__int64 a1)
{
  __int64 v1; // r9
  __int64 v2; // r9
  __int64 v3; // rdx
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // r8
  unsigned __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = 8 * v1;
    v3 = 0;
    do
    {
      v9 = *(_QWORD *)(a1 + 8);
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + v3);
      if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
        v4 = *(_QWORD *)(v9 - 8);
      else
        v4 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
      v5 = (_QWORD *)(3 * v3 + v4);
      if ( *v5 )
      {
        v6 = v5[1];
        v7 = v5[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v7 = v6;
        if ( v6 )
          *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
      }
      *v5 = v10;
      if ( v10 )
      {
        v8 = *(_QWORD *)(v10 + 8);
        v5[1] = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = (unsigned __int64)(v5 + 1) | *(_QWORD *)(v8 + 16) & 3LL;
        v5[2] = (v10 + 8) | v5[2] & 3LL;
        *(_QWORD *)(v10 + 8) = v5;
      }
      v3 += 8;
    }
    while ( v2 != v3 );
  }
}
