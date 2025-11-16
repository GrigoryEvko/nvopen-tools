// Function: sub_1240EC0
// Address: 0x1240ec0
//
__int64 __fastcall sub_1240EC0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // rsi
  unsigned __int8 v6; // [rsp+Fh] [rbp-91h]
  __int64 v7; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v8; // [rsp+20h] [rbp-80h] BYREF
  __int64 v9; // [rsp+28h] [rbp-78h]
  _BYTE v10[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a1 + 232);
  result = sub_120AFE0(a1, 407, "expected uselistorder directive");
  if ( !(_BYTE)result )
  {
    v5 = (__int64)&v7;
    v8 = v10;
    v9 = 0x1000000000LL;
    if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v7, a2)
      || (v5 = 4, (unsigned __int8)sub_120AFE0(a1, 4, "expected comma in uselistorder directive"))
      || (v5 = (__int64)&v8, sub_1210710(a1, (__int64)&v8)) )
    {
      result = 1;
    }
    else
    {
      v5 = v7;
      result = sub_123FE40(a1, v7, (__int64)v8, (unsigned int)v9, v3);
    }
    if ( v8 != v10 )
    {
      v6 = result;
      _libc_free(v8, v5);
      return v6;
    }
  }
  return result;
}
