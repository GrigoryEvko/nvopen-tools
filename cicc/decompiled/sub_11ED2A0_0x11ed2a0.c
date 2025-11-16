// Function: sub_11ED2A0
// Address: 0x11ed2a0
//
__int64 __fastcall sub_11ED2A0(__int64 **a1, __int64 a2, __int64 a3, int a4)
{
  __int64 *v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 result; // rax
  __int64 v13; // [rsp+10h] [rbp-30h]
  __int64 v14; // [rsp+18h] [rbp-28h]

  BYTE4(v14) = 0;
  BYTE4(v13) = 0;
  if ( !sub_11EC990((__int64)a1, a2, 3u, 0x100000002LL, v13, v14) )
    return 0;
  v7 = *a1;
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v9 = *(_QWORD *)(a2 + 32 * (2 - v8));
  v10 = *(_QWORD *)(a2 + 32 * (1 - v8));
  v11 = *(_QWORD *)(a2 - 32 * v8);
  if ( a4 != 153 )
  {
    result = sub_11CA400(v11, v10, v9, a3, v7);
    if ( result )
      goto LABEL_4;
    return 0;
  }
  result = sub_11CA350(v11, v10, v9, a3, v7);
  if ( !result )
    return 0;
LABEL_4:
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
