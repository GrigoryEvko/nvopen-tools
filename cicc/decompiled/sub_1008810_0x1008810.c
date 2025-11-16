// Function: sub_1008810
// Address: 0x1008810
//
__int64 __fastcall sub_1008810(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  bool v6; // zf
  __int64 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v8; // [rsp+8h] [rbp-18h] BYREF

  if ( *a1 > 0x15u || (result = sub_96E680(12, (__int64)a1)) == 0 )
  {
    v8 = &v7;
    v6 = (unsigned __int8)sub_995E90(&v8, (unsigned __int64)a1, a3, a4, a5) == 0;
    result = 0;
    if ( !v6 )
      return v7;
  }
  return result;
}
