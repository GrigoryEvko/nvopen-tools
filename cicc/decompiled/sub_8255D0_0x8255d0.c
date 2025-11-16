// Function: sub_8255D0
// Address: 0x8255d0
//
void __fastcall sub_8255D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  char v7; // al
  _BOOL8 v8; // rsi
  _BOOL8 v9; // r8
  _BOOL4 v10; // eax
  char v11; // al

  if ( (*(_BYTE *)(a1 + 198) & 0x20) != 0
    || (*(_BYTE *)(a1 + 193) & 0x10) == 0
    && ((*(_QWORD *)(a1 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(a1 + 192) & 2) != 0) )
  {
    return;
  }
  if ( (*(_BYTE *)(a1 + 197) & 0x60) != 0 )
    return;
  if ( !qword_4F04C50 )
    return;
  v6 = *(_QWORD *)(qword_4F04C50 + 32LL);
  if ( !v6 || *(_BYTE *)(a1 + 174) == 6 || (*(_BYTE *)(a1 + 206) & 2) != 0 )
    return;
  if ( (*(_BYTE *)(v6 + 193) & 0x10) == 0
    && ((*(_QWORD *)(v6 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(v6 + 192) & 2) != 0) )
  {
    v7 = *(_BYTE *)(v6 + 198);
    v8 = (v7 & 0x10) != 0;
    if ( (v7 & 8) == 0 )
    {
      v9 = (v7 & 0x10) == 0;
      v10 = (v7 & 0x10) != 0 || (v7 & 0x10) == 0;
      goto LABEL_12;
    }
LABEL_22:
    v9 = 1;
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(v6 + 197) & 0x60) != 0 )
    return;
  v11 = *(_BYTE *)(v6 + 198);
  v8 = (v11 & 0x10) != 0;
  if ( (v11 & 8) != 0 )
    goto LABEL_22;
  v10 = (v11 & 0x10) != 0;
  v9 = 0;
LABEL_12:
  if ( v10 )
LABEL_13:
    sub_825360(v9, v8, 0, a1, v9, a6);
}
