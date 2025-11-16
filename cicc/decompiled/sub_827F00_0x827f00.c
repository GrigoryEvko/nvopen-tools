// Function: sub_827F00
// Address: 0x827f00
//
__int64 __fastcall sub_827F00(__int64 a1, __int64 a2, unsigned int a3, int a4, int a5, _DWORD *a6)
{
  unsigned int v6; // r11d
  __int64 v10; // rbx
  __int64 v11; // r12
  char v12; // al
  __int64 v13; // rsi
  _DWORD *v15; // r15
  char v16; // di
  _QWORD *v17; // rax
  __int64 v18; // rdi
  _BOOL4 v19; // eax
  __int64 v20; // rax
  unsigned int v21; // [rsp+8h] [rbp-48h]
  unsigned int v24; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a3;
  v10 = a2;
  v11 = *(_QWORD *)a1;
  v25[0] = 0;
  if ( a6 )
    *a6 = 0;
  while ( *(_BYTE *)(v11 + 140) == 12 )
    v11 = *(_QWORD *)(v11 + 160);
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
      v10 = *(_QWORD *)(v10 + 160);
    while ( *(_BYTE *)(v10 + 140) == 12 );
  }
  v12 = *(_BYTE *)(a1 + 16);
  if ( v12 == 2 )
  {
    v13 = a1 + 144;
    v25[0] = a1 + 144;
    goto LABEL_10;
  }
  if ( *(_BYTE *)(a1 + 17) == 1 )
  {
    v19 = sub_6ED0A0(a1);
    v6 = a3;
    if ( !v19 )
    {
      v20 = sub_6ED2B0(a1);
      v6 = a3;
      v25[0] = v20;
      v13 = v20;
      goto LABEL_10;
    }
    v12 = *(_BYTE *)(a1 + 16);
  }
  v13 = v25[0];
  if ( v12 != 1 )
  {
LABEL_10:
    if ( !(unsigned int)sub_8D67E0(v11, v13, v10, v6, &v24) || (unsigned int)sub_696840(a1) )
      return 0;
    goto LABEL_13;
  }
  v21 = v6;
  v17 = sub_724DC0();
  v18 = *(_QWORD *)(a1 + 144);
  v25[0] = v17;
  if ( !(unsigned int)sub_719770(v18, (__int64)v17, 0, 0) )
  {
    sub_724E30((__int64)v25);
    v13 = v25[0];
    v6 = v21;
    goto LABEL_10;
  }
  if ( !(unsigned int)sub_8D67E0(v11, v25[0], v10, v21, &v24) || (unsigned int)sub_696840(a1) )
  {
    sub_724E30((__int64)v25);
    return 0;
  }
  sub_724E30((__int64)v25);
LABEL_13:
  if ( a5 )
  {
    if ( v24 == 2362 && (unsigned int)sub_8D2780(v11) && (unsigned int)sub_8D2780(v10) )
      return 0;
LABEL_15:
    v15 = (_DWORD *)(a1 + 68);
    v16 = 5;
    if ( !a4 )
    {
LABEL_16:
      sub_6E5D70(v16, v24, v15, v11, v10);
      return 1;
    }
    if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
    {
      if ( !sub_67D3C0((int *)v24, 7, v15) )
        goto LABEL_25;
    }
    else if ( !sub_67D370((int *)v24, 7u, v15) )
    {
LABEL_25:
      *a6 = 1;
      return 0;
    }
    v16 = 7;
    goto LABEL_16;
  }
  if ( a4 )
    goto LABEL_15;
  return 1;
}
