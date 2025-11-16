// Function: sub_11ECE30
// Address: 0x11ece30
//
unsigned __int8 *__fastcall sub_11ECE30(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned __int8 *v5; // rdi
  __int64 v7; // [rsp+10h] [rbp-30h]
  __int64 v8; // [rsp+18h] [rbp-28h]

  BYTE4(v8) = 0;
  BYTE4(v7) = 0;
  v4 = sub_B43CC0(a2);
  if ( sub_11EC990((__int64)a1, a2, 3u, 0x100000002LL, v7, v8)
    && (v5 = (unsigned __int8 *)sub_11CA6D0(
                                  *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                                  *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                                  *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                                  a3,
                                  v4,
                                  *a1)) != 0 )
  {
    return sub_11DAF00(v5, a2);
  }
  else
  {
    return 0;
  }
}
