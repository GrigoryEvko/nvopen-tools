// Function: sub_2B172D0
// Address: 0x2b172d0
//
__int64 __fastcall sub_2B172D0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-30h] BYREF
  __int64 v6; // [rsp+18h] [rbp-20h]

  sub_2B16B50(
    (unsigned __int64 *)&v5,
    *(__int64 ***)(*(_QWORD *)a1 + 3296LL),
    a2,
    a3,
    a4,
    **(unsigned __int8 **)(*(_QWORD *)(a1 + 8) + 416LL) - 29,
    *(_DWORD *)(a1 + 16),
    *(_QWORD *)(a1 + 24),
    *(_QWORD *)(a1 + 32));
  result = v6 - v5;
  if ( __OFSUB__(v6, v5) )
  {
    result = 0x8000000000000000LL;
    if ( v5 <= 0 )
      return 0x7FFFFFFFFFFFFFFFLL;
  }
  return result;
}
