// Function: sub_24EE180
// Address: 0x24ee180
//
__int64 __fastcall sub_24EE180(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  _BOOL4 v4; // r12d
  _QWORD *v5; // rax
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 280) = sub_24E5090(
                            *(_QWORD *)(a1 + 8),
                            v1,
                            *(const __m128i **)(a1 + 16),
                            (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL) + 24LL),
                            *(_QWORD *)(a1 + 296));
  sub_24EA1C0(a1, v1, v2, v3);
  v4 = *(_DWORD *)(a1 + 32) == 2;
  v7[0] = *(_QWORD *)(**(_QWORD **)(a1 + 24) - 32LL * (*(_DWORD *)(**(_QWORD **)(a1 + 24) + 4LL) & 0x7FFFFFF));
  v5 = sub_24E84F0(a1 + 200, v7);
  return sub_24F3340(v5[2], v4);
}
