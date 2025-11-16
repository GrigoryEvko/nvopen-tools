// Function: sub_1F03240
// Address: 0x1f03240
//
__int64 __fastcall sub_1F03240(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v4; // rax
  int v5; // edx
  int v6; // r14d
  __int64 result; // rax
  __int64 v8; // rdx
  _BYTE v9[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_DWORD *)(v4 + 4LL * *(unsigned int *)(a3 + 192));
  v6 = *(_DWORD *)(v4 + 4LL * *(unsigned int *)(a2 + 192));
  v9[0] = 0;
  result = 0;
  if ( v5 < v6 )
  {
    v8 = *(_QWORD *)(a1 + 72);
    if ( v8 )
      memset(*(void **)(a1 + 64), 0, 8 * v8);
    sub_1F03050(a1, a3, v6, v9);
    return v9[0];
  }
  return result;
}
