// Function: sub_7ADC90
// Address: 0x7adc90
//
__int64 __fastcall sub_7ADC90(int a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  char v4; // al
  __int64 v5; // rdx
  FILE *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned int v12; // eax

  v1 = unk_4D04A18;
  if ( !unk_4D04A18 )
    return 0;
  v2 = 0;
  if ( (*(_BYTE *)(unk_4D04A18 + 81LL) & 0x10) != 0 )
    v2 = **(_QWORD **)(unk_4D04A18 + 64LL);
  if ( (unk_4D04A11 & 0x20) != 0 )
    return 0;
  if ( (unk_4D04A10 & 1) == 0 )
    return 0;
  v4 = *(_BYTE *)(unk_4D04A18 + 80LL);
  if ( v4 == 19 )
    return 0;
  if ( v4 == 3 )
  {
    if ( *(_BYTE *)(unk_4D04A18 + 104LL) )
    {
      v8 = *(_QWORD *)(unk_4D04A18 + 88LL);
      if ( (*(_BYTE *)(v8 + 177) & 0x10) != 0 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v8 + 168) + 168LL) )
          return 0;
      }
    }
  }
  else
  {
    if ( (unsigned __int8)(v4 - 20) <= 1u )
      return 0;
    if ( ((v4 - 7) & 0xFD) == 0 )
    {
      v5 = *(_QWORD *)(unk_4D04A18 + 88LL);
      if ( !v5 )
        goto LABEL_16;
      if ( (*(_BYTE *)(v5 + 170) & 0x10) != 0 && **(_QWORD **)(v5 + 216) )
        return 0;
    }
    if ( v4 == 17 && (unsigned int)sub_8780F0(unk_4D04A18) )
      return 0;
  }
LABEL_16:
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
    return 0;
  if ( (*(_BYTE *)(v1 + 81) & 0x10) == 0 )
  {
    if ( (a1 & 0x2000) == 0 )
      return 0;
LABEL_28:
    sub_6854C0(0x317u, (FILE *)dword_4F07508, v1);
    return 1;
  }
  if ( (unsigned __int8)(*(_BYTE *)(v2 + 80) - 4) <= 1u && *(char *)(*(_QWORD *)(v2 + 88) + 177LL) < 0 )
    return 0;
  if ( (a1 & 0x2000) != 0 )
    goto LABEL_28;
  if ( (a1 & 0x100000) != 0 )
    return 0;
  v6 = *(FILE **)(v1 + 64);
  v7 = *(_QWORD *)&v6->_flags;
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)&v6->_flags + 80LL) - 4) > 1u )
  {
LABEL_22:
    sub_685360(0x1D0u, dword_4F07508, (__int64)v6);
    return 1;
  }
  v9 = *(_BYTE *)(*(_QWORD *)(v7 + 88) + 177LL);
  if ( v9 < 0 )
    return 0;
  if ( (v9 & 0x30) != 0x30 )
    goto LABEL_22;
  v10 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 72LL);
  v11 = sub_7ADC30(v10);
  if ( *(_QWORD *)(v11 + 88) && (*(_BYTE *)(v11 + 160) & 1) == 0 )
    v10 = *(_QWORD *)(v11 + 88);
  if ( *(_QWORD *)(*(_QWORD *)(v10 + 88) + 176LL) )
  {
    sub_6851C0(0x1F2u, dword_4F07508);
  }
  else
  {
    v12 = sub_67F240();
    sub_685A50(v12, dword_4F07508, v6, 8u);
  }
  return 1;
}
