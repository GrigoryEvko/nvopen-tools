// Function: sub_38B7E70
// Address: 0x38b7e70
//
__int64 __fastcall sub_38B7E70(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  unsigned __int64 v5; // rbx
  __int64 result; // rax
  unsigned __int8 v7; // [rsp+Fh] [rbp-91h]
  __int64 v8; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v9; // [rsp+20h] [rbp-80h] BYREF
  __int64 v10; // [rsp+28h] [rbp-78h]
  _BYTE v11[112]; // [rsp+30h] [rbp-70h] BYREF

  v5 = *(_QWORD *)(a1 + 56);
  result = sub_388AF10(a1, 301, "expected uselistorder directive");
  if ( !(_BYTE)result )
  {
    v9 = v11;
    v10 = 0x1000000000LL;
    if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v8, a2, a3, a4, a5)
      || (unsigned __int8)sub_388AF10(a1, 4, "expected comma in uselistorder directive")
      || (unsigned __int8)sub_388EBB0(a1, (__int64)&v9) )
    {
      result = 1;
    }
    else
    {
      result = sub_38B6F20(a1, v8, (__int64)v9, (unsigned int)v10, v5);
    }
    if ( v9 != v11 )
    {
      v7 = result;
      _libc_free((unsigned __int64)v9);
      return v7;
    }
  }
  return result;
}
