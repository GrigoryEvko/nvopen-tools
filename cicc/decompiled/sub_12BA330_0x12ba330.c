// Function: sub_12BA330
// Address: 0x12ba330
//
__int64 __fastcall sub_12BA330(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  unsigned int v6; // r12d

  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v5 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( a1 )
    {
      if ( a2 )
      {
        *(_QWORD *)(a1 + 216) = a3;
        *(_QWORD *)(a1 + 208) = a2;
        v6 = 0;
      }
      else
      {
        v6 = 4;
      }
    }
    else
    {
      v6 = 5;
    }
    sub_16C30E0(v5);
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( a1 )
    {
      if ( a2 )
      {
        *(_QWORD *)(a1 + 208) = a2;
        *(_QWORD *)(a1 + 216) = a3;
        return 0;
      }
      return 4;
    }
    else
    {
      return 5;
    }
  }
  return v6;
}
