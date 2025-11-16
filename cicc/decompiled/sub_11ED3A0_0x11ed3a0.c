// Function: sub_11ED3A0
// Address: 0x11ed3a0
//
__int64 __fastcall sub_11ED3A0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+10h] [rbp-30h]
  __int64 v7; // [rsp+18h] [rbp-28h]

  BYTE4(v7) = 0;
  BYTE4(v6) = 0;
  if ( !sub_11EC990((__int64)a1, a2, 4u, 0x100000003LL, v6, v7) )
    return 0;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  result = sub_11CAA80(
             *(_QWORD *)(a2 - 32 * v4),
             *(_QWORD *)(a2 + 32 * (1 - v4)),
             *(_QWORD *)(a2 + 32 * (2 - v4)),
             *(_QWORD *)(a2 + 32 * (3 - v4)),
             a3,
             *a1);
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
