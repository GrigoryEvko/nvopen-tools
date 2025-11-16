// Function: sub_11E3340
// Address: 0x11e3340
//
__int64 __fastcall sub_11E3340(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r8
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v9[0] = 0x100000000LL;
  sub_11DAA90(a2, (int *)v9, 2, v5, v4);
  v6 = *(_QWORD *)(a2 - 32);
  if ( v6 && !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
    return 0;
  v7 = (unsigned __int8 *)sub_B343C0(
                            a3,
                            0xEEu,
                            *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                            0x100u,
                            *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
                            0x100u,
                            v5,
                            0,
                            0,
                            0,
                            0,
                            0);
  sub_11DAF00(v7, a2);
  return *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
}
