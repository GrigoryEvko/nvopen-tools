// Function: sub_11E3150
// Address: 0x11e3150
//
__int64 __fastcall sub_11E3150(__int64 a1, __int64 a2, unsigned int **a3)
{
  unsigned __int64 v4; // r8
  __int64 v5; // rdx
  unsigned __int8 *v6; // r13
  unsigned __int8 *v7; // r14
  __int64 v8; // r15
  __int64 result; // rax
  __int64 v10; // rcx
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(unsigned __int8 **)(a2 - 32 * v5);
  v7 = *(unsigned __int8 **)(a2 + 32 * (1 - v5));
  v8 = *(_QWORD *)(a2 + 32 * (2 - v5));
  v11[0] = 0x100000000LL;
  sub_11DAA90(a2, (int *)v11, 2, v8, v4);
  result = sub_11DCF00(a2, (__int64)v6, (__int64)v7, v8, 0, a3);
  if ( !result && *(_BYTE *)v8 == 17 )
  {
    v10 = *(_QWORD *)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      v10 = **(_QWORD **)(v8 + 24);
    return sub_11DC340(a2, v6, v7, v10, (__int64)a3, *(_QWORD *)(a1 + 16));
  }
  return result;
}
