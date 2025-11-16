// Function: sub_18E71D0
// Address: 0x18e71d0
//
__int64 __fastcall sub_18E71D0(__int64 a1, unsigned int a2, __int64 a3)
{
  char v5; // di
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rcx
  unsigned int v13; // r8d
  __int64 v14; // rcx
  _QWORD *v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx

  v5 = *(_BYTE *)(a1 + 23) & 0x40;
  if ( *(_BYTE *)(a1 + 16) != 77 )
  {
LABEL_15:
    if ( v5 )
      v6 = *(_QWORD *)(a1 - 8);
    else
      v6 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
LABEL_17:
    v16 = (_QWORD *)(v6 + 24LL * a2);
    if ( *v16 )
    {
      v17 = v16[1];
      v18 = v16[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v18 = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
    }
    *v16 = a3;
    v13 = 1;
    if ( a3 )
    {
      v19 = *(_QWORD *)(a3 + 8);
      v16[1] = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v19 + 16) & 3LL;
      v13 = 1;
      v16[2] = (a3 + 8) | v16[2] & 3LL;
      *(_QWORD *)(a3 + 8) = v16;
    }
    return v13;
  }
  if ( v5 )
    v6 = *(_QWORD *)(a1 - 8);
  else
    v6 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( !a2 )
    goto LABEL_17;
  v8 = 0;
  while ( 1 )
  {
    v7 = v6 + 24LL * *(unsigned int *)(a1 + 56) + 8;
    if ( *(_QWORD *)(v7 + 8 * v8) == *(_QWORD *)(v7 + 8LL * a2) )
      break;
    if ( a2 <= (unsigned int)++v8 )
      goto LABEL_15;
  }
  v9 = *(_QWORD *)(v6 + 24 * v8);
  v10 = (_QWORD *)(v6 + 24LL * a2);
  if ( *v10 )
  {
    v11 = v10[1];
    v12 = v10[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *v10 = v9;
  v13 = 0;
  if ( !v9 )
    return v13;
  v14 = *(_QWORD *)(v9 + 8);
  v10[1] = v14;
  if ( v14 )
    *(_QWORD *)(v14 + 16) = (unsigned __int64)(v10 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
  v10[2] = (v9 + 8) | v10[2] & 3LL;
  *(_QWORD *)(v9 + 8) = v10;
  return 0;
}
