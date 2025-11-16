// Function: sub_131DE10
// Address: 0x131de10
//
__int64 __fastcall sub_131DE10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 result; // rax

  sub_1317140(
    a1,
    a3,
    (_DWORD *)(a2 + 24),
    (_QWORD *)(a2 + 32),
    (__int64 *)(a2 + 40),
    (__int64 *)(a2 + 48),
    a2 + 56,
    a2 + 64,
    a2 + 72,
    *(_QWORD *)(a2 + 80),
    *(_QWORD *)(a2 + 80) + 10416LL,
    *(_QWORD *)(a2 + 80) + 15600LL,
    *(_QWORD *)(a2 + 80) + 25008LL,
    *(_QWORD *)(a2 + 80) + 34560LL,
    *(_QWORD *)(a2 + 80) + 37760LL);
  v4 = 0;
  do
  {
    v5 = *(_QWORD *)(a2 + 80);
    v6 = (_QWORD *)(v5 + 144LL * (unsigned int)v4);
    v7 = qword_505FA40[v4++] * v6[1305];
    *(_QWORD *)(v5 + 10368) += v7;
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 10376LL) += v6[1302];
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 10384LL) += v6[1303];
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 10392LL) += v6[1304];
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 10400LL) += v6[1306];
    result = v6[1307];
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 10408LL) += result;
  }
  while ( v4 != 36 );
  return result;
}
