// Function: sub_12BA560
// Address: 0x12ba560
//
__int64 __fastcall sub_12BA560(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned int v6; // r12d

  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v4 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( a1 )
    {
      v5 = 1;
      if ( *(_QWORD *)(a1 + 56) )
        v5 = *(_QWORD *)(a1 + 56);
      if ( a2 )
      {
        *a2 = v5;
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
    sub_16C30E0(v4);
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( a1 )
    {
      v2 = 1;
      if ( *(_QWORD *)(a1 + 56) )
        v2 = *(_QWORD *)(a1 + 56);
      if ( a2 )
      {
        *a2 = v2;
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
