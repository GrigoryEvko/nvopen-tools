// Function: sub_649830
// Address: 0x649830
//
_BYTE *__fastcall sub_649830(__int64 a1, __int64 a2, int a3)
{
  unsigned __int8 v6; // bl
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rcx
  int v11; // edx
  _BYTE *result; // rax
  int v13; // r15d
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v6 = *(_BYTE *)(a1 + 80) - 10;
  v7 = *(_QWORD *)(a1 + 88);
  if ( v6 <= 1u )
  {
    v8 = *(_QWORD *)(v7 + 152);
    LOBYTE(v11) = (*(_BYTE *)(v7 + 206) & 4) != 0;
    if ( (*(_BYTE *)(v7 + 193) & 0x10) != 0 )
    {
      v10 = v7;
      goto LABEL_4;
    }
    v11 = (unsigned __int8)v11;
    v10 = v7;
    v9 = 0;
  }
  else
  {
    v8 = *(_QWORD *)(v7 + 120);
    v9 = v7;
    v10 = 0;
    v11 = (*(_BYTE *)(v7 + 175) & 0x10) != 0;
  }
  if ( (*(_BYTE *)(v7 + 88) & 0x50) == 0x10
    || dword_4F04C44 != -1
    || (result = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64), (result[6] & 2) != 0) )
  {
LABEL_4:
    result = 0;
    v13 = 0;
    if ( !a3 )
      return result;
LABEL_5:
    if ( dword_4D04438 )
      return result;
    goto LABEL_6;
  }
  if ( !(a3 | v11) )
    return result;
  v19 = v10;
  v13 = sub_8D96C0();
  result = (_BYTE *)sub_8D9700(v8);
  v10 = v19;
  if ( a3 )
  {
    if ( (unsigned int)result | v13 )
    {
      if ( v6 <= 1u )
        *(_BYTE *)(v19 + 206) |= 4u;
      else
        *(_BYTE *)(v9 + 175) |= 0x10u;
    }
    else
    {
      result = 0;
      v13 = 0;
    }
    goto LABEL_5;
  }
LABEL_6:
  v20 = v10;
  if ( !v13 )
  {
    if ( !(_DWORD)result )
      return result;
    if ( dword_4D04438 )
    {
      v14 = a1;
      v15 = a2;
      v16 = 1772;
      return (_BYTE *)sub_6853B0(7, v16, v15, v14);
    }
    if ( dword_4D04964 )
    {
      v17 = unk_4F07471;
      v18 = (unsigned int)(v6 < 2u) + 961;
    }
    else
    {
      v17 = 5 - (unsigned int)(v6 >= 2u);
      v18 = 962 - (unsigned int)(v6 >= 2u);
    }
    return (_BYTE *)sub_684AA0(v17, v18, a2);
  }
  result = (_BYTE *)sub_8DD010(v8);
  if ( (_DWORD)result )
    return result;
  result = &dword_4D04438;
  if ( !dword_4D04438 )
  {
    if ( qword_4D0495C )
    {
      v17 = 5;
    }
    else
    {
      if ( dword_4F077BC && v6 > 1u )
      {
        v17 = 5;
        v18 = 544;
        return (_BYTE *)sub_684AA0(v17, v18, a2);
      }
      v17 = 8;
    }
    v18 = (unsigned int)(v6 < 2u) + 544;
    return (_BYTE *)sub_684AA0(v17, v18, a2);
  }
  if ( !v20 || (*(_WORD *)(v20 + 192) & 0x4003) != 0 && (*(_BYTE *)(v20 + 192) & 8) == 0 )
  {
    v14 = a1;
    v15 = a2;
    v16 = 1771;
    return (_BYTE *)sub_6853B0(7, v16, v15, v14);
  }
  return result;
}
