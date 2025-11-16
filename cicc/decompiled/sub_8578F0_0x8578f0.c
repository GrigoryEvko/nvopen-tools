// Function: sub_8578F0
// Address: 0x8578f0
//
__int64 __fastcall sub_8578F0(__int64 a1, unsigned int *a2, unsigned int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r12d
  __int64 result; // rax

  v7 = a4;
  if ( dword_4D04944 )
  {
    if ( !a1 || *(_DWORD *)(a1 + 12) != 5 )
    {
      if ( !(_DWORD)a4 )
        goto LABEL_5;
      return sub_7BE1A0(a1, (__int64)a2, (__int64)a3, a4, a5, a6);
    }
  }
  else if ( !a1 )
  {
    a2 = a3;
    a1 = 161;
    sub_684B30(0xA1u, a3);
    if ( !v7 )
    {
LABEL_5:
      for ( result = (unsigned int)word_4F06418[0] - 9;
            (unsigned __int16)(word_4F06418[0] - 9) > 1u;
            result = (unsigned int)word_4F06418[0] - 9 )
      {
        sub_7B8B50(a1, a2, (__int64)a3, a4, a5, a6);
      }
      return result;
    }
    return sub_7BE1A0(a1, (__int64)a2, (__int64)a3, a4, a5, a6);
  }
  result = (__int64)sub_856400(a1, a2, a3, a4, a5);
  if ( *(_BYTE *)(a1 + 8) == 39 )
  {
    result = qword_4F04C50;
    if ( qword_4F04C50 )
    {
      result = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( result )
      {
        if ( (*(_BYTE *)(result + 198) & 0x10) != 0 )
          return sub_684B30(0xE48u, a3);
      }
    }
  }
  return result;
}
