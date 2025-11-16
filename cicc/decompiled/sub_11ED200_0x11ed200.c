// Function: sub_11ED200
// Address: 0x11ed200
//
__int64 __fastcall sub_11ED200(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+18h] [rbp-28h]

  BYTE4(v8) = 0;
  BYTE4(v7) = 0;
  if ( !sub_11EC990((__int64)a1, a2, 1u, v7, 0x100000000LL, v8) )
    return 0;
  v4 = *a1;
  v5 = sub_B43CC0(a2);
  result = sub_11CA050(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a3, v5, v4);
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
