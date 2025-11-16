// Function: sub_25B4BE0
// Address: 0x25b4be0
//
__int64 __fastcall sub_25B4BE0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r13
  int v5; // ebx
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 i; // r13
  unsigned __int64 v13; // rsi
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // rdx

  v3 = a3 + 24;
  v5 = 0;
  v6 = *(_QWORD *)(a3 + 32);
  if ( v6 == a3 + 24 )
  {
    sub_25B3CB0(a2);
    v10 = *(_QWORD *)(a3 + 32);
    if ( v6 == v10 )
    {
      v15 = a1 + 32;
      v16 = a1 + 80;
      goto LABEL_21;
    }
    goto LABEL_11;
  }
  do
  {
    while ( 1 )
    {
      v7 = v6;
      v6 = *(_QWORD *)(v6 + 8);
      if ( *(_DWORD *)(*(_QWORD *)(v7 - 32) + 8LL) >> 8 )
        break;
      if ( v3 == v6 )
        goto LABEL_6;
    }
    v5 |= sub_25AFA00((__int64)a2, v7 - 56);
  }
  while ( v3 != v6 );
LABEL_6:
  v8 = *(_QWORD *)(a3 + 32);
  if ( v3 != v8 )
  {
    do
    {
      v9 = v8 - 56;
      if ( !v8 )
        v9 = 0;
      sub_25B41B0(a2, v9);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v6 != v8 );
  }
  sub_25B3CB0(a2);
  v10 = *(_QWORD *)(a3 + 32);
  if ( v6 != v10 )
  {
    do
    {
LABEL_11:
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 8);
      v5 |= sub_25B1010(a2, (char *)(v11 - 56));
    }
    while ( v3 != v10 );
    for ( i = *(_QWORD *)(a3 + 32); v10 != i; LOBYTE(v5) = v14 | v5 )
    {
      v13 = i - 56;
      if ( !i )
        v13 = 0;
      v14 = sub_25AFA50((__int64)a2, v13);
      i = *(_QWORD *)(i + 8);
    }
  }
  v15 = a1 + 32;
  v16 = a1 + 80;
  if ( !(_BYTE)v5 )
  {
LABEL_21:
    *(_QWORD *)(a1 + 56) = v16;
    *(_QWORD *)(a1 + 8) = v15;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v15;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v16;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
