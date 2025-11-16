// Function: sub_5CB1C0
// Address: 0x5cb1c0
//
__int64 __fastcall sub_5CB1C0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v7; // [rsp+8h] [rbp-48h] BYREF
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a2;
  v3 = sub_5C7B50(a1, (__int64)&v7, a3);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 32);
    v5 = v3;
    if ( v4 )
    {
      if ( *(_BYTE *)(v4 + 10) )
      {
        do
        {
          while ( !(unsigned int)sub_5CACA0(v4, a1, 1, 2147483646, v8) )
          {
            v4 = *(_QWORD *)v4;
            if ( !v4 )
              return v7;
          }
          sub_5C95A0(*(__int64 ***)(v5 + 168), v8[0], a1 + 56);
          v4 = *(_QWORD *)v4;
        }
        while ( v4 );
      }
    }
    else
    {
      sub_5C95A0(*(__int64 ***)(v3 + 168), 0, a1 + 56);
    }
  }
  return v7;
}
