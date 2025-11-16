// Function: sub_37373C0
// Address: 0x37373c0
//
void __fastcall sub_37373C0(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rax
  unsigned __int8 *v9; // r13
  char v10; // al
  unsigned __int8 v11; // al
  _BYTE *v12; // rsi
  __int64 v13; // rdx

  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) == 0 )
  {
    v5 = a2 - 16 - 8LL * ((v4 >> 2) & 0xF);
    v6 = *(_QWORD *)(v5 + 48);
    if ( v6 )
      goto LABEL_3;
LABEL_11:
    v9 = *(unsigned __int8 **)(v5 + 8);
    goto LABEL_6;
  }
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_QWORD *)(v5 + 48);
  if ( !v6 )
    goto LABEL_11;
LABEL_3:
  v7 = *(_BYTE *)(v6 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(v6 - 32);
  else
    v8 = v6 - 16 - 8LL * ((v7 >> 2) & 0xF);
  v9 = *(unsigned __int8 **)(v8 + 8);
LABEL_6:
  v10 = sub_3736590(a1);
  sub_324FBC0((__int64)a1, a2, a3, v10);
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) == 0 )
  {
    v12 = *(_BYTE **)(a2 - 8LL * ((v11 >> 2) & 0xF));
    if ( v12 )
      goto LABEL_8;
LABEL_14:
    v13 = 0;
    goto LABEL_9;
  }
  v12 = *(_BYTE **)(*(_QWORD *)(a2 - 32) + 16LL);
  if ( !v12 )
    goto LABEL_14;
LABEL_8:
  v12 = (_BYTE *)sub_B91420((__int64)v12);
LABEL_9:
  sub_3736650((__int64)a1, v12, v13, a3, v9);
}
