// Function: sub_12BA220
// Address: 0x12ba220
//
__int64 __fastcall sub_12BA220(__int64 a1, _QWORD *a2)
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
      *a2 = *(_QWORD *)(a1 + 88) + 1LL;
      v4 = 0;
    }
    else
    {
      v4 = 5;
    }
    sub_16C30E0(v3);
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( a1 )
    {
      *a2 = *(_QWORD *)(a1 + 88) + 1LL;
      return 0;
    }
    return 5;
  }
  return v4;
}
