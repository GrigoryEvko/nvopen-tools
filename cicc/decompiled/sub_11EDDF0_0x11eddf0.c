// Function: sub_11EDDF0
// Address: 0x11eddf0
//
__int64 __fastcall sub_11EDDF0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-38h]
  __int64 v6; // [rsp+10h] [rbp-30h]

  BYTE4(v6) = 0;
  BYTE4(v5) = 0;
  if ( !sub_11EC990((__int64)a1, a2, 2u, v5, v6, 0x100000001LL) )
    return 0;
  result = sub_11CB120(
             *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
             *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             *(_QWORD *)(a2 + 32 * (4LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             a3,
             *a1);
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
