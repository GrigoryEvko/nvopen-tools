// Function: sub_7E52E0
// Address: 0x7e52e0
//
__int64 __fastcall sub_7E52E0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rax
  const __m128i *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9

  v1 = sub_7E51A0(*(_QWORD *)(a1 + 56));
  v2 = sub_73E830(v1);
  v3 = (const __m128i *)sub_73DCD0(v2);
  if ( (*(_BYTE *)(a1 + 25) & 1) == 0 )
    v3 = (const __m128i *)sub_731370((__int64)v3, (__int64)v3, v4, v5, v6, v7);
  return sub_730620(a1, v3);
}
