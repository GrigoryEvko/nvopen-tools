// Function: sub_7E3BF0
// Address: 0x7e3bf0
//
__int64 __fastcall sub_7E3BF0(__int64 a1, _DWORD *a2, __int64 *a3, int *a4)
{
  __int64 v6; // rax
  _BYTE *v7; // r15
  char v8; // al
  __int64 result; // rax
  int v10; // edx
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  char v18; // cl

  v6 = sub_7E3BE0(a1);
  *a2 = 0;
  v7 = (_BYTE *)v6;
  *a3 = 0;
  v8 = *(_BYTE *)(a1 + 88);
  if ( (v8 & 0x70) != 0x20 )
  {
    *a2 = 1;
    result = (v8 & 4) != 0;
    goto LABEL_3;
  }
  v12 = *(_QWORD *)(a1 + 168);
  v13 = *(_QWORD *)(v12 + 152);
  if ( !v13 || (*(_BYTE *)(v13 + 29) & 0x20) != 0 || (*(_WORD *)(a1 + 176) & 0x110) == 0 )
    goto LABEL_9;
  if ( (*(_BYTE *)(v12 + 110) & 0x20) == 0 )
  {
    v15 = *(_BYTE *)(a1 + 178);
    if ( (v15 & 8) == 0 )
    {
      if ( (*(_DWORD *)(a1 + 176) & 0x11000) == 0x1000 )
      {
        if ( (v15 & 0x10) == 0 )
        {
LABEL_28:
          v10 = 1;
          result = 1;
          goto LABEL_10;
        }
LABEL_9:
        v10 = 0;
        result = 0;
        goto LABEL_10;
      }
      v16 = sub_735B60(a1, 0);
      v17 = v16;
      if ( v16 )
      {
        *a3 = v16;
        result = *(_DWORD *)(v16 + 160) != 0;
        v18 = *(_BYTE *)(v17 + 88) & 0x70;
        v10 = 0;
        if ( v18 == 16 )
        {
          *a2 = 1;
LABEL_3:
          v10 = 0;
          if ( !v7 )
            goto LABEL_20;
LABEL_4:
          if ( (v7[156] & 0x20) == 0 )
            goto LABEL_11;
          goto LABEL_5;
        }
      }
      else
      {
        if ( unk_4D048C8 != 2 )
        {
          if ( unk_4D048C8 != 1 && (*(_BYTE *)(a1 + 178) & 0x10) == 0 )
            goto LABEL_28;
          goto LABEL_9;
        }
        v10 = 0;
        result = 1;
      }
LABEL_10:
      if ( !v7 )
        goto LABEL_11;
      goto LABEL_4;
    }
  }
  *a2 = 0;
  if ( !v7 )
  {
    *a4 = 0;
    return 1;
  }
  result = 1;
  v10 = 0;
  if ( (v7[156] & 0x20) == 0 )
    goto LABEL_12;
LABEL_5:
  if ( !HIDWORD(qword_4D045BC) )
  {
    *a2 = 1;
    v11 = result;
    v10 = 1;
    goto LABEL_13;
  }
LABEL_11:
  if ( !*a2 )
  {
LABEL_12:
    v11 = result & v10;
    goto LABEL_13;
  }
LABEL_20:
  v11 = result;
  v10 = 1;
LABEL_13:
  *a4 = v10;
  if ( v11 )
  {
    if ( !dword_4D03F94 )
      return v11;
    v7[174] |= 1u;
    v14 = *(_QWORD *)(a1 + 152);
    if ( v14 )
    {
      if ( (*(_BYTE *)(v14 + 88) & 4) != 0 && !unk_4D03F88 )
        return v11;
    }
    if ( (v7[88] & 4) != 0 )
      return v11;
    else
      return (*(_BYTE *)(a1 + 88) & 0x70) != 32;
  }
  return result;
}
