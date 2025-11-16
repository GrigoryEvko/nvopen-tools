// Function: sub_12140B0
// Address: 0x12140b0
//
__int64 __fastcall sub_12140B0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned __int8 v5; // al
  _QWORD *v6; // r9
  __int64 v7; // [rsp+0h] [rbp-70h]
  unsigned __int8 v8; // [rsp+8h] [rbp-68h]
  __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  __int64 v10; // [rsp+18h] [rbp-58h]
  __int64 v11; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-48h]
  __int64 v13; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-38h]

  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_1211570(a1, a2)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_12115D0(a1, (__int64)(a2 + 1)) )
  {
    return 1;
  }
  if ( *(_DWORD *)(a1 + 240) == 4 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( !(unsigned __int8)sub_120AFE0(a1, 439, "expected 'calls' here")
      && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
      && !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    {
      do
      {
        v9 = 0;
        v10 = 0;
        sub_AADB10((__int64)&v11, 0x40u, 1);
        v5 = sub_1213C20(a1, &v9, a3);
        if ( v5 )
        {
          v8 = v5;
          sub_969240(&v13);
          sub_969240(&v11);
          return v8;
        }
        v6 = (_QWORD *)a2[6];
        if ( v6 == (_QWORD *)a2[7] )
        {
          sub_1213D80(a2 + 5, a2[6], (__int64)&v9);
        }
        else
        {
          if ( v6 )
          {
            v7 = a2[6];
            *v6 = v9;
            v6[1] = v10;
            sub_9865C0((__int64)(v6 + 2), (__int64)&v11);
            sub_9865C0(v7 + 32, (__int64)&v13);
          }
          a2[6] += 48;
        }
        if ( v14 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        if ( v12 > 0x40 && v11 )
          j_j___libc_free_0_0(v11);
      }
      while ( *(_DWORD *)(a1 + 240) == 4 && (unsigned __int8)sub_1205540(a1) );
      if ( !(unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
        return sub_120AFE0(a1, 13, "expected ')' here");
    }
    return 1;
  }
  return sub_120AFE0(a1, 13, "expected ')' here");
}
