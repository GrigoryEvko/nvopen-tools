// Function: sub_1218580
// Address: 0x1218580
//
__int64 __fastcall sub_1218580(__int64 a1, __int64 **a2, unsigned int a3)
{
  unsigned int v3; // r13d
  int v5; // eax
  unsigned __int64 v7; // [rsp+8h] [rbp-78h]
  int v8; // [rsp+1Ch] [rbp-64h]
  _QWORD v9[4]; // [rsp+20h] [rbp-60h] BYREF
  char v10; // [rsp+40h] [rbp-40h]
  char v11; // [rsp+41h] [rbp-3Fh]

  v3 = 0;
  sub_A74A00((__int64)a2);
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = *(_DWORD *)(a1 + 240);
        if ( v5 != 512 )
          break;
        if ( (unsigned __int8)sub_120C270(a1, a2) )
          return 1;
      }
      if ( v5 != 273 )
        break;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      sub_A77CE0(a2, 0);
    }
    if ( (unsigned int)(v5 - 166) > 0x61 )
      return v3;
    v8 = v5 - 165;
    v7 = *(_QWORD *)(a1 + 232);
    if ( (unsigned __int8)sub_1217C30(a1, v5 - 165, a2, 0) )
      break;
    if ( (_BYTE)a3 )
    {
      if ( !sub_A71A10(v8) )
      {
        v11 = 1;
        v3 = a3;
        v9[0] = "this attribute does not apply to parameters";
        v10 = 3;
        sub_11FD800(a1 + 176, v7, (__int64)v9, 1);
      }
    }
    else if ( !sub_A71A30(v8) )
    {
      v11 = 1;
      v3 = 1;
      v9[0] = "this attribute does not apply to return values";
      v10 = 3;
      sub_11FD800(a1 + 176, v7, (__int64)v9, 1);
    }
  }
  return 1;
}
