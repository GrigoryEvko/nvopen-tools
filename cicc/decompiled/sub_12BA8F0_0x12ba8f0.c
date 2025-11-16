// Function: sub_12BA8F0
// Address: 0x12ba8f0
//
__int64 __fastcall sub_12BA8F0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // r13
  unsigned int v4; // r12d

  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v3 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( a1 )
    {
      if ( !sub_2241570(a1 + 48, a2, *(_QWORD *)(a1 + 56), 0) )
        *a2 = 0;
      v4 = 0;
    }
    else
    {
      v4 = 5;
    }
    sub_16C30E0(v3);
    return v4;
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( a1 )
    {
      if ( !sub_2241570(a1 + 48, a2, *(_QWORD *)(a1 + 56), 0) )
        *a2 = 0;
      return 0;
    }
    else
    {
      return 5;
    }
  }
}
