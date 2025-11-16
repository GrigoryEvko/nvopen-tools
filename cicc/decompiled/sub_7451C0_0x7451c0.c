// Function: sub_7451C0
// Address: 0x7451c0
//
__int64 __fastcall sub_7451C0(signed __int64 a1, __int64 (__fastcall **a2)(char *, _QWORD))
{
  char v3; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v4[79]; // [rsp+1h] [rbp-4Fh] BYREF

  if ( a1 < 0 )
  {
    v3 = 45;
    if ( a1 < -9 )
    {
      sub_622470(-a1, v4);
    }
    else
    {
      v4[1] = 0;
      v4[0] = 48 - a1;
    }
  }
  else if ( a1 > 9 )
  {
    sub_622470(a1, &v3);
  }
  else
  {
    v4[0] = 0;
    v3 = a1 + 48;
  }
  return (*a2)(&v3, a2);
}
