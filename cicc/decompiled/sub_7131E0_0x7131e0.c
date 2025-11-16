// Function: sub_7131E0
// Address: 0x7131e0
//
__int64 __fastcall sub_7131E0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 result; // rax
  char i; // al
  __int64 v6; // r13

  *a3 = 0;
  result = *(unsigned __int8 *)(a1 + 173);
  if ( (_BYTE)result == 1 )
  {
    for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(a2 + 140) )
      a2 = *(_QWORD *)(a2 + 160);
    if ( i == 15 )
      v6 = *(_QWORD *)(*(_QWORD *)(a2 + 160) + 128LL) * dword_4F06BA0;
    else
      v6 = *(_QWORD *)(a2 + 128) * dword_4F06BA0;
    result = sub_6210B0(a1, 0);
    if ( (int)result >= 0 )
    {
      result = sub_6210B0(a1, v6);
      if ( (int)result >= 0 )
        *a3 = 63;
    }
    else
    {
      *a3 = 62;
    }
  }
  else if ( (_BYTE)result != 8 )
  {
    sub_721090(a1);
  }
  return result;
}
