// Function: sub_A59DB0
// Address: 0xa59db0
//
void __fastcall sub_A59DB0(__int64 a1, __int64 a2)
{
  char v3; // al
  _BYTE *v4; // rsi
  _BYTE *v5; // rsi
  _BYTE *v6; // rsi
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_BYTE *)(a2 + 32);
  if ( v3 )
  {
    if ( v3 != 1 )
      BUG();
    v6 = *(_BYTE **)(a2 + 40);
    goto LABEL_12;
  }
  v4 = *(_BYTE **)(a2 + 40);
  if ( (unsigned __int8)(*v4 - 5) <= 0x1Fu )
    sub_A59AF0(a1, v4);
  sub_A59AF0(a1, *(_BYTE **)(a2 + 72));
  if ( *(_BYTE *)(a2 + 64) == 2 )
  {
    sub_A59AF0(a1, *(_BYTE **)(a2 + 56));
    v6 = *(_BYTE *)(a2 + 64) == 2 ? *(_BYTE **)(a2 + 48) : *(_BYTE **)(a2 + 40);
    if ( (unsigned __int8)(*v6 - 5) <= 0x1Fu )
LABEL_12:
      sub_A59AF0(a1, v6);
  }
  v5 = *(_BYTE **)(a2 + 24);
  v7[0] = v5;
  if ( v5 )
  {
    sub_B96E90(v7, v5, 1);
    v5 = (_BYTE *)v7[0];
  }
  sub_A59AF0(a1, v5);
  if ( v7[0] )
    sub_B91220(v7);
}
