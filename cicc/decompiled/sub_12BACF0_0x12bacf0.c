// Function: sub_12BACF0
// Address: 0x12bacf0
//
__int64 __fastcall sub_12BACF0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // r12d
  __int64 v7; // r15

  if ( !byte_4F92D70 && dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    if ( !a1 )
      return 5;
    if ( a4 == 61453 )
    {
      a1[16] = a2;
      a1[17] = a3;
      return 0;
    }
    if ( a4 > 0xF00D )
    {
      if ( a4 == 64222 )
      {
        a1[14] = a2;
        a1[15] = a3;
        return 0;
      }
    }
    else
    {
      if ( a4 == 47710 )
      {
        a1[20] = a2;
        a1[21] = a3;
        return 0;
      }
      if ( a4 == 56993 )
      {
        a1[18] = a2;
        a1[19] = a3;
        return 0;
      }
    }
    return 0;
  }
  if ( !qword_4F92D80 )
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  v7 = qword_4F92D80;
  sub_16C30C0(qword_4F92D80);
  if ( a1 )
  {
    if ( a4 == 61453 )
    {
      a1[16] = a2;
      a1[17] = a3;
    }
    else if ( a4 > 0xF00D )
    {
      if ( a4 == 64222 )
      {
        a1[14] = a2;
        a1[15] = a3;
      }
    }
    else if ( a4 == 47710 )
    {
      a1[20] = a2;
      a1[21] = a3;
    }
    else if ( a4 == 56993 )
    {
      a1[18] = a2;
      a1[19] = a3;
    }
    v6 = 0;
  }
  else
  {
    v6 = 5;
  }
  sub_16C30E0(v7);
  return v6;
}
