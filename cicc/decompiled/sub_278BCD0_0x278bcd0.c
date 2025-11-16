// Function: sub_278BCD0
// Address: 0x278bcd0
//
_BYTE *__fastcall sub_278BCD0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // r9
  unsigned int v7; // esi
  _DWORD *v8; // r14
  int v9; // edi
  __int64 v10; // r12
  _QWORD *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // r15
  _BYTE *v14; // r13
  __int64 v15; // rax
  unsigned int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // r11d
  __int64 v21; // [rsp+0h] [rbp-40h]
  bool v22; // [rsp+Fh] [rbp-31h]

  v3 = *(unsigned int *)(a1 + 376);
  v4 = *(_QWORD *)(a1 + 360);
  if ( !(_DWORD)v3 )
    return 0;
  v7 = (v3 - 1) & (37 * a3);
  v8 = (_DWORD *)(v4 + 40LL * v7);
  v9 = *v8;
  if ( a3 != *v8 )
  {
    v20 = 1;
    while ( v9 != -1 )
    {
      v7 = (v3 - 1) & (v20 + v7);
      v8 = (_DWORD *)(v4 + 40LL * v7);
      v9 = *v8;
      if ( a3 == *v8 )
        goto LABEL_3;
      ++v20;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (_DWORD *)(v4 + 40 * v3) )
    return 0;
  v10 = *(_QWORD *)(a1 + 24);
  v11 = v8 + 2;
  if ( a2 )
  {
    v12 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    if ( (unsigned int)(*(_DWORD *)(a2 + 44) + 1) < *(_DWORD *)(v10 + 32) )
    {
LABEL_6:
      v13 = *(_QWORD *)(*(_QWORD *)(v10 + 24) + 8 * v12);
      v22 = v13 == 0;
      goto LABEL_7;
    }
  }
  else
  {
    v12 = 0;
    if ( *(_DWORD *)(v10 + 32) )
      goto LABEL_6;
  }
  v22 = 1;
  v13 = 0;
LABEL_7:
  v14 = 0;
  v15 = sub_278BC90(a1, (__int64)v11);
  if ( v15 == v13 )
    goto LABEL_18;
  while ( v22 )
  {
    do
    {
LABEL_18:
      v14 = (_BYTE *)*v11;
      if ( *(_BYTE *)*v11 <= 0x15u )
        return v14;
LABEL_16:
      v11 = (_QWORD *)v11[3];
      if ( !v11 )
        return v14;
      v10 = *(_QWORD *)(a1 + 24);
      v15 = sub_278BC90(a1, (__int64)v11);
    }
    while ( v15 == v13 );
  }
  if ( !v15 )
    goto LABEL_16;
  if ( v15 == *(_QWORD *)(v13 + 8) )
    goto LABEL_18;
  if ( v13 == *(_QWORD *)(v15 + 8) || *(_DWORD *)(v15 + 16) >= *(_DWORD *)(v13 + 16) )
    goto LABEL_16;
  if ( *(_BYTE *)(v10 + 112) )
  {
    if ( *(_DWORD *)(v13 + 72) < *(_DWORD *)(v15 + 72) )
      goto LABEL_16;
LABEL_15:
    if ( *(_DWORD *)(v13 + 76) <= *(_DWORD *)(v15 + 76) )
      goto LABEL_18;
    goto LABEL_16;
  }
  v17 = *(_DWORD *)(v10 + 116) + 1;
  *(_DWORD *)(v10 + 116) = v17;
  if ( v17 > 0x20 )
  {
    v21 = v15;
    sub_B19440(v10);
    v15 = v21;
    if ( *(_DWORD *)(v13 + 72) < *(_DWORD *)(v21 + 72) )
      goto LABEL_16;
    goto LABEL_15;
  }
  v18 = v13;
  do
  {
    v19 = v18;
    v18 = *(_QWORD *)(v18 + 8);
  }
  while ( v18 && *(_DWORD *)(v15 + 16) <= *(_DWORD *)(v18 + 16) );
  if ( v15 != v19 )
    goto LABEL_16;
  v14 = (_BYTE *)*v11;
  if ( *(_BYTE *)*v11 > 0x15u )
    goto LABEL_16;
  return v14;
}
