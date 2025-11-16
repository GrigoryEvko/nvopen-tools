// Function: sub_6418E0
// Address: 0x6418e0
//
__int64 __fastcall sub_6418E0(__int64 a1)
{
  __int64 result; // rax
  char v3; // dl
  __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx

  *(_BYTE *)(a1 + 85) = 0;
  result = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)result != 2 )
  {
    if ( (_DWORD)result == 1 )
      *(_BYTE *)(a1 + 84) = 1;
    return result;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( (*(_BYTE *)(a1 + 65) & 8) == 0 )
    {
      *(_BYTE *)(a1 + 84) = 3;
      goto LABEL_8;
    }
LABEL_7:
    *(_BYTE *)(a1 + 84) = 2;
LABEL_8:
    result = *(_QWORD *)(a1 + 56);
    if ( *(_BYTE *)(result + 140) != 7 )
      return result;
    v3 = *(_BYTE *)(a1 + 84);
LABEL_10:
    v4 = *(_QWORD *)(result + 168);
    result = *(_BYTE *)(v4 + 17) & 0x8F;
    *(_BYTE *)(v4 + 17) = *(_BYTE *)(v4 + 17) & 0x8F | (16 * (v3 & 7));
    return result;
  }
  if ( (*(_WORD *)(a1 + 64) & 0x802) != 0 )
    goto LABEL_7;
  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 )
  {
    if ( (*(_BYTE *)(v5 + 64) & 0x20) != 0 )
      goto LABEL_7;
  }
  v6 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9);
  if ( (v6 & 0x10) == 0
    || dword_4F077BC && (*(_BYTE *)(a1 + 64) & 0x10) != 0 && *(_QWORD *)(a1 + 24) && (*(_BYTE *)(a1 + 65) & 1) == 0 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( !v7 )
    {
      v7 = *(_QWORD *)(a1 + 24);
      if ( !v7 )
        goto LABEL_31;
    }
    v8 = *(unsigned __int8 *)(v7 + 80);
    if ( (_BYTE)v8 == 16 )
    {
      v7 = **(_QWORD **)(v7 + 88);
      v8 = *(unsigned __int8 *)(v7 + 80);
    }
    if ( (_BYTE)v8 == 24 )
    {
      v7 = *(_QWORD *)(v7 + 88);
      v8 = *(unsigned __int8 *)(v7 + 80);
    }
    if ( ((unsigned __int8)v8 > 0x12u || (v9 = 270716, !_bittest64(&v9, v8))) && (v5 == 0) == ((_BYTE)v8 == 7) )
      *(_BYTE *)(a1 + 84) = (*(_BYTE *)(sub_87D520(v7) + 88) >> 4) & 7;
    else
LABEL_31:
      *(_BYTE *)(a1 + 84) = (v6 >> 1) & 7;
  }
  else
  {
    *(_BYTE *)(a1 + 85) = 1;
    *(_BYTE *)(a1 + 84) = (v6 >> 1) & 7;
  }
  result = *(_QWORD *)(a1 + 56);
  if ( *(_BYTE *)(result + 140) == 7 )
  {
    v3 = *(_BYTE *)(a1 + 84);
    if ( (unsigned __int8)(v3 - 2) <= 1u )
      goto LABEL_10;
  }
  return result;
}
