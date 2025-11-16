// Function: sub_728620
// Address: 0x728620
//
__int64 __fastcall sub_728620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r8d
  int v8; // eax
  int v9; // eax

  if ( (unsigned int)sub_8D2660(a1) )
  {
    if ( !(unsigned int)sub_8D2660(a3) && !(unsigned int)sub_8D2EF0(a3) )
    {
      v6 = sub_8D3D10(a3);
      if ( !v6 )
      {
        if ( a4 )
        {
          v9 = sub_8D26D0(a1);
          v6 = 0;
          if ( v9 )
            return sub_712570(a4);
        }
        return v6;
      }
    }
    return 1;
  }
  if ( (unsigned int)sub_8D2EF0(a1) )
    return 1;
  v6 = sub_8D3D10(a1);
  if ( v6 )
    return 1;
  if ( a2 )
  {
    v8 = sub_8D26D0(a3);
    v6 = 0;
    if ( v8 )
      return sub_712570(a2);
  }
  return v6;
}
