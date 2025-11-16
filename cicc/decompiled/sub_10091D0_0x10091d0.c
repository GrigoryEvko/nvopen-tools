// Function: sub_10091D0
// Address: 0x10091d0
//
unsigned __int8 *__fastcall sub_10091D0(__int64 *a1, _BYTE *a2, char a3, __int64 *a4, char a5, char a6)
{
  __int64 v8; // r12
  _BYTE *v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v11; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v12; // [rsp+10h] [rbp-30h] BYREF
  _BYTE *v13; // [rsp+18h] [rbp-28h]

  v11 = a1;
  v10 = a2;
  if ( a5 || a6 != 1 )
  {
    v12 = v11;
    v13 = v10;
    return sub_1003820((__int64 *)&v12, 2, a3, (__int64)a4, a5, a6);
  }
  else
  {
    v8 = sub_FFE3E0(0x18u, (_BYTE **)&v11, &v10, a4);
    if ( !v8 )
    {
      v12 = v11;
      v13 = v10;
      v8 = (__int64)sub_1003820((__int64 *)&v12, 2, a3, (__int64)a4, 0, 1);
      if ( !v8 && (a3 & 2) != 0 )
      {
        v12 = 0;
        if ( (unsigned __int8)sub_10069D0(&v12, (__int64)v11) )
        {
          return sub_AD9290(v11[1], 0);
        }
        else
        {
          v12 = 0;
          if ( (unsigned __int8)sub_1008640(&v12, (__int64)v11) )
            return sub_AD9290(v11[1], 1);
        }
      }
    }
  }
  return (unsigned __int8 *)v8;
}
