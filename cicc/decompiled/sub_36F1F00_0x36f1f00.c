// Function: sub_36F1F00
// Address: 0x36f1f00
//
__int64 __fastcall sub_36F1F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rdx
  char v9; // r14
  __int64 v11; // rdx
  __int64 v12; // rcx
  char v13; // [rsp+Fh] [rbp-31h]

  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, a2, a3, a4);
    v5 = *(_QWORD *)(a3 + 96);
    v6 = v5 + 40LL * *(_QWORD *)(a3 + 104);
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a3, a2, v11, v12);
      v5 = *(_QWORD *)(a3 + 96);
    }
  }
  else
  {
    v5 = *(_QWORD *)(a3 + 96);
    v6 = v5 + 40LL * *(_QWORD *)(a3 + 104);
  }
  v7 = a1 + 32;
  v8 = a1 + 80;
  if ( v6 == v5 )
    goto LABEL_15;
  v13 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) == 14 )
      {
        v9 = sub_B2D680(v5);
        if ( v9 )
        {
          if ( !(unsigned __int8)sub_CE8660(v5) || !(unsigned __int8)sub_CE9220(a3) )
            break;
        }
      }
      v5 += 40;
      if ( v5 == v6 )
        goto LABEL_10;
    }
    sub_36F1C60(a3, v5);
    v13 = v9;
    v5 += 40;
  }
  while ( v5 != v6 );
LABEL_10:
  v7 = a1 + 32;
  v8 = a1 + 80;
  if ( !v13 )
  {
LABEL_15:
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v8;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v7;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v8;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  return a1;
}
