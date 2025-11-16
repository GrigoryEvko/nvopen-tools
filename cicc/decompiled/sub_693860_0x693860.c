// Function: sub_693860
// Address: 0x693860
//
_BOOL8 __fastcall sub_693860(unsigned __int64 a1, int a2, int a3, _DWORD *a4)
{
  _BOOL4 v5; // r12d
  __int64 v8; // rax
  int v9; // esi
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  *a4 = 0;
  if ( *(_BYTE *)(a1 + 136) <= 2u )
  {
    *a4 = 1731;
    return 0;
  }
  if ( (unsigned int)sub_6935B0(a1, -1, 0) != -1 )
  {
    v5 = sub_68B790(a1);
    if ( v5 )
    {
      *a4 = 1586;
      return 0;
    }
    if ( (unsigned int)sub_8DD010(*(_QWORD *)(a1 + 120)) )
    {
      if ( !(dword_4F077BC | (unsigned int)qword_4F077B4)
        || !a3
        || !(unsigned int)sub_8D4070(*(_QWORD *)(a1 + 120))
        || (v8 = sub_8D4050(*(_QWORD *)(a1 + 120)), (unsigned int)sub_8DD010(v8)) )
      {
        *a4 = 1756;
        return v5;
      }
    }
    if ( sub_6879B0() )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) != 0 )
      {
LABEL_16:
        *a4 = 394;
        return v5;
      }
    }
    else
    {
      v9 = -1;
      while ( 1 )
      {
        v9 = sub_6935B0(a1, v9, v10);
        if ( !v10[0] )
          break;
        if ( (*(_BYTE *)(*(_QWORD *)(**(_QWORD **)(v10[0] + 8LL) + 96LL) + 181LL) & 2) != 0 )
          goto LABEL_16;
      }
    }
    v5 = 1;
    if ( (*(_BYTE *)(a1 + 170) & 1) == 0 )
      return v5;
    if ( dword_4F077C4 == 2 )
    {
      if ( unk_4F07778 > 202001 )
        return v5;
      if ( (_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A0 > 0x270FFu )
        {
LABEL_28:
          *a4 = 3357;
          return 1;
        }
LABEL_24:
        *a4 = 2850;
        return 0;
      }
    }
    else if ( (_DWORD)qword_4F077B4 )
    {
      goto LABEL_24;
    }
    if ( dword_4F077BC && qword_4F077A8 > 0x1116Fu )
    {
      if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
        return 1;
      goto LABEL_28;
    }
    goto LABEL_24;
  }
  *a4 = (a2 == 0) + 1736;
  return 0;
}
