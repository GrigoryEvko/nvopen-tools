// Function: sub_832D70
// Address: 0x832d70
//
__int64 __fastcall sub_832D70(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  bool v3; // zf
  _QWORD *v4; // r12

  result = *(unsigned __int8 *)(a1 + 9);
  if ( (result & 1) == 0 )
    goto LABEL_12;
  v2 = *(_BYTE *)(a1 + 8);
  result = (unsigned int)result & 0xFFFFFFFE;
  *(_BYTE *)(a1 + 9) = result;
  if ( !v2 )
  {
    if ( dword_4F077C4 != 2
      || (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0
      || (v4 = *(_QWORD **)(a1 + 32), *(_QWORD *)(a1 + 32) = 0, !v4) )
    {
      result = *(unsigned __int8 *)(a1 + 9);
      if ( (result & 2) == 0 )
      {
LABEL_5:
        if ( (result & 0x40) != 0 )
        {
          *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x20u;
          result = *(unsigned __int8 *)(a1 + 9);
        }
        goto LABEL_7;
      }
LABEL_17:
      sub_6E18E0(*(_QWORD *)(a1 + 24) + 8LL);
      result = *(_BYTE *)(a1 + 9) & 0xFD;
      v3 = *(_BYTE *)(a1 + 8) == 0;
      *(_BYTE *)(a1 + 9) &= ~2u;
      if ( !v3 )
        goto LABEL_7;
      goto LABEL_5;
    }
    sub_734370(v4);
    sub_732E20((__int64)v4);
    result = *(unsigned __int8 *)(a1 + 9);
LABEL_12:
    if ( (result & 2) == 0 )
    {
      if ( *(_BYTE *)(a1 + 8) )
        goto LABEL_7;
      goto LABEL_5;
    }
    goto LABEL_17;
  }
  if ( v2 == 1 )
  {
    sub_832E80(*(_QWORD *)(a1 + 24));
    result = *(unsigned __int8 *)(a1 + 9);
    goto LABEL_12;
  }
  if ( v2 != 2 )
    sub_721090();
  if ( (result & 2) != 0 )
    goto LABEL_17;
LABEL_7:
  if ( (result & 0x80u) != 0LL )
  {
    result = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 20LL) |= 4u;
  }
  return result;
}
