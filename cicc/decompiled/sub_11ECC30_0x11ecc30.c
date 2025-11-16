// Function: sub_11ECC30
// Address: 0x11ecc30
//
__int64 __fastcall sub_11ECC30(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r8
  __int64 result; // rax
  unsigned __int8 *v6; // rax
  __int64 v7; // [rsp+10h] [rbp-20h]
  __int64 v8; // [rsp+18h] [rbp-18h]

  BYTE4(v8) = 0;
  BYTE4(v7) = 0;
  v4 = sub_11EC990(a1, a2, 3u, 0x100000002LL, v7, v8);
  result = 0;
  if ( v4 )
  {
    v6 = (unsigned __int8 *)sub_B343C0(
                              a3,
                              0xF1u,
                              *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                              0x100u,
                              *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                              0x100u,
                              *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                              0,
                              0,
                              0,
                              0,
                              0);
    sub_11DAF00(v6, a2);
    return *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  return result;
}
