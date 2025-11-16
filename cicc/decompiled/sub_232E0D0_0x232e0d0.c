// Function: sub_232E0D0
// Address: 0x232e0d0
//
__int64 __fastcall sub_232E0D0(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+0h] [rbp-30h] BYREF
  __int64 v4; // [rsp+8h] [rbp-28h]
  __int64 v5; // [rsp+18h] [rbp-18h] BYREF

  v3 = a1;
  v4 = a2;
  BYTE4(v5) = (unsigned __int8)sub_95CB50((const void **)&v3, "devirt<", 7u)
           && (unsigned __int8)sub_232E070(&v3, ">", 1u)
           && !sub_C93CC0(v3, v4, 0, &v5)
           && v5 == (int)v5
           && (int)v5 >= 0;
  return v5;
}
