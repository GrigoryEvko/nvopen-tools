// Function: sub_30D6950
// Address: 0x30d6950
//
__int64 __fastcall sub_30D6950(__int64 a1, int a2)
{
  int v2; // ecx
  __int64 v4; // rdi
  _QWORD *v6; // r14
  _QWORD *v7; // r13
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // eax
  int v18; // eax
  int v19; // eax
  int v20; // eax
  int v21; // eax
  int v23; // eax

  v2 = a1 + 8;
  v4 = a1 + 16;
  *(_QWORD *)(v4 - 8) = 0;
  *(_QWORD *)(v4 + 60) = 0;
  memset((void *)(v4 & 0xFFFFFFFFFFFFFFF8LL), 0, 8LL * ((v2 - ((unsigned int)v4 & 0xFFFFFFF8) + 76) >> 3));
  *(_DWORD *)(a1 + 64) = 65792;
  v6 = sub_C52410();
  v7 = v6 + 1;
  v8 = sub_C959E0();
  v9 = (_QWORD *)v6[2];
  if ( v9 )
  {
    v10 = v6 + 1;
    do
    {
      while ( 1 )
      {
        v11 = v9[2];
        v12 = v9[3];
        if ( v8 <= v9[4] )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v10 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v11 );
LABEL_6:
    if ( v7 != v10 && v8 >= v10[4] )
      v7 = v10;
  }
  if ( v7 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v13 = v7[7];
    if ( v13 )
    {
      v14 = v7 + 6;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v13 + 16);
          v16 = *(_QWORD *)(v13 + 24);
          if ( *(_DWORD *)(v13 + 32) >= dword_5030B68 )
            break;
          v13 = *(_QWORD *)(v13 + 24);
          if ( !v16 )
            goto LABEL_15;
        }
        v14 = (_QWORD *)v13;
        v13 = *(_QWORD *)(v13 + 16);
      }
      while ( v15 );
LABEL_15:
      if ( v7 + 6 != v14 && dword_5030B68 >= *((_DWORD *)v14 + 8) && *((int *)v14 + 9) > 0 )
        a2 = qword_5030BE8;
    }
  }
  v17 = dword_5030B08;
  *(_BYTE *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 40) = 1;
  *(_DWORD *)(a1 + 4) = v17;
  v18 = dword_50304E8;
  *(_DWORD *)a1 = a2;
  *(_DWORD *)(a1 + 36) = v18;
  if ( (int)sub_23DF0D0(dword_5030388) > 0 )
  {
    v19 = dword_5030408;
    *(_BYTE *)(a1 + 48) = 1;
    *(_DWORD *)(a1 + 44) = v19;
  }
  v20 = dword_5030A28;
  *(_BYTE *)(a1 + 56) = 1;
  *(_DWORD *)(a1 + 52) = v20;
  if ( !(unsigned int)sub_23DF0D0(&dword_5030B68) )
  {
    v21 = dword_50305C8;
    *(_BYTE *)(a1 + 32) = 1;
    *(_DWORD *)(a1 + 28) = 5;
    *(_DWORD *)(a1 + 20) = 50;
    *(_BYTE *)(a1 + 24) = 1;
    *(_DWORD *)(a1 + 12) = v21;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  if ( (int)sub_23DF0D0(dword_5030548) <= 0 )
    return a1;
  v23 = dword_50305C8;
  *(_BYTE *)(a1 + 16) = 1;
  *(_DWORD *)(a1 + 12) = v23;
  return a1;
}
