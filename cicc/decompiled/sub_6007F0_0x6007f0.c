// Function: sub_6007F0
// Address: 0x6007f0
//
__int64 __fastcall sub_6007F0(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  unsigned int v4; // r15d
  __int64 v5; // r14
  char v6; // r13
  char v7; // r12
  __int64 v8; // rdi
  char v9; // al
  __int64 i; // rdi
  int v12; // eax
  _BYTE v13[5]; // [rsp+7h] [rbp-39h]

  v3 = *(_QWORD *)(a1 + 160);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u || (*(_DWORD *)(a1 + 176) & 0x11000) != 0x1000 )
    sub_5E87D0(a1);
  v4 = 1;
  v5 = sub_72FD90(v3, 7);
  if ( !v5 )
    return v4;
  v6 = *(_BYTE *)(a1 + 140);
  v13[4] = 0;
  *(_DWORD *)v13 = v6 == 11;
  v7 = (a2 ^ 1) & 1;
  while ( (*(_BYTE *)(v5 + 146) & 1) == 0 || !v7 )
  {
    v8 = *(_QWORD *)(v5 + 120);
    if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 && (sub_8D4C10(v8, dword_4F077C4 != 2) & 2) != 0 && v7 )
      break;
    v9 = *(_BYTE *)(v5 + 144);
    if ( (v9 & 0x40) != 0 )
    {
      if ( (v9 & 0x10) == 0 )
        goto LABEL_7;
      for ( i = *(_QWORD *)(v5 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v12 = sub_6007F0(i, a2);
      if ( v6 == 11 )
      {
LABEL_28:
        if ( !v12 )
          goto LABEL_7;
        if ( !*(_DWORD *)&v13[1] )
        {
LABEL_30:
          *(_DWORD *)&v13[1] = 1;
          goto LABEL_7;
        }
        goto LABEL_15;
      }
      goto LABEL_25;
    }
    if ( (*(_BYTE *)(v5 + 145) & 0x20) == 0 )
    {
      v12 = sub_600680(*(_QWORD *)(v5 + 120), a1) != 0;
      if ( v6 == 11 )
        goto LABEL_28;
LABEL_25:
      if ( !v12 )
        return 0;
      goto LABEL_7;
    }
    if ( v6 == 11 )
    {
      if ( !*(_DWORD *)&v13[1] )
        goto LABEL_30;
LABEL_15:
      v4 = 0;
    }
LABEL_7:
    v5 = sub_72FD90(*(_QWORD *)(v5 + 112), 7);
    if ( !v5 )
      goto LABEL_19;
  }
  v4 = 0;
LABEL_19:
  if ( !*(_DWORD *)&v13[1] && v6 == 11 )
    return 0;
  return v4;
}
