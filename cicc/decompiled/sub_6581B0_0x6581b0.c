// Function: sub_6581B0
// Address: 0x6581b0
//
__int64 __fastcall sub_6581B0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a2 + 8);
  if ( (result & 2) != 0 )
  {
    sub_658080((_BYTE *)a1, a3);
    result = *(_QWORD *)(a2 + 8);
  }
  if ( (result & 0x80000) != 0 )
  {
    if ( a3 )
      goto LABEL_5;
    if ( (*(_BYTE *)(a1 + 176) & 1) != 0 )
      goto LABEL_5;
    if ( (*(_BYTE *)(a1 + 170) & 0x20) != 0 )
    {
      result = sub_8DD3B0(*(_QWORD *)(a1 + 120));
      if ( (_DWORD)result )
        goto LABEL_5;
      result = sub_8D3A70(*(_QWORD *)(a1 + 120));
      if ( (_DWORD)result )
        goto LABEL_5;
    }
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( !qword_4F077A8 )
          goto LABEL_15;
        goto LABEL_12;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_15;
    }
    if ( qword_4F077A0 <= 0x78B3u || dword_4F077C4 != 2 || unk_4F07778 <= 201702 )
      goto LABEL_15;
LABEL_12:
    if ( (*(_BYTE *)(a2 + 130) & 4) != 0 )
    {
      result = sub_684B30(2385, a2 + 112);
LABEL_5:
      *(_BYTE *)(a1 + 172) |= 8u;
      return result;
    }
LABEL_15:
    sub_6851C0(2385, a2 + 112);
    result = sub_72C930(2385);
    *(_QWORD *)(a1 + 120) = result;
  }
  return result;
}
