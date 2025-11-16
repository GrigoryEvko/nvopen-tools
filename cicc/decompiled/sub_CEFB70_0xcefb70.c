// Function: sub_CEFB70
// Address: 0xcefb70
//
__int64 __fastcall sub_CEFB70(__int64 a1)
{
  unsigned int v1; // r13d
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  bool v4; // zf
  _QWORD *v6; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v7[10]; // [rsp+10h] [rbp-50h] BYREF

  v2 = *(_BYTE **)(a1 + 232);
  v3 = *(_QWORD *)(a1 + 240);
  v6 = v7;
  sub_CEF010((__int64 *)&v6, v2, (__int64)&v2[v3]);
  v4 = *(_DWORD *)(a1 + 276) == 21;
  v7[2] = *(_QWORD *)(a1 + 264);
  LOBYTE(v1) = v4;
  v7[3] = *(_QWORD *)(a1 + 272);
  v7[4] = *(_QWORD *)(a1 + 280);
  if ( v6 != v7 )
    j_j___libc_free_0(v6, v7[0] + 1LL);
  return v1;
}
