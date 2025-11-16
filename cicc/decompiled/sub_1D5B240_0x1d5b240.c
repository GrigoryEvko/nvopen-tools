// Function: sub_1D5B240
// Address: 0x1d5b240
//
void __fastcall sub_1D5B240(__int64 a1)
{
  unsigned int *v1; // rdx
  unsigned int *i; // r8
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rsi

  v1 = *(unsigned int **)(a1 + 16);
  for ( i = &v1[4 * *(unsigned int *)(a1 + 24)]; v1 != i; v1 += 4 )
  {
    v9 = *(_QWORD *)v1;
    v10 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(*(_QWORD *)v1 + 23LL) & 0x40) != 0 )
      v4 = *(_QWORD *)(v9 - 8);
    else
      v4 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    v5 = (_QWORD *)(v4 + 24LL * v1[2]);
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
  }
}
