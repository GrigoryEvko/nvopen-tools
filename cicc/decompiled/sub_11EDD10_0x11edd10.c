// Function: sub_11EDD10
// Address: 0x11edd10
//
__int64 __fastcall sub_11EDD10(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+10h] [rbp-30h]

  BYTE4(v6) = 0;
  if ( !sub_11EC990((__int64)a1, a2, 3u, 0x100000001LL, v6, 0x100000002LL) )
    return 0;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  result = sub_11CB040(
             *(_QWORD *)(a2 - 32 * v4),
             *(_QWORD *)(a2 + 32 * (1 - v4)),
             *(_QWORD *)(a2 + 32 * (4 - v4)),
             *(_QWORD *)(a2 + 32 * (5 - v4)),
             a3,
             *a1);
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
