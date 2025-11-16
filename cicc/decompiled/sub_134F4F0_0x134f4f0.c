// Function: sub_134F4F0
// Address: 0x134f4f0
//
__int64 __fastcall sub_134F4F0(_QWORD *a1, __int64 a2, const __m128i *a3)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v6; // rdx

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 48);
  v3 &= 0xFFFFFFFFFFFFFFF8LL;
  v6 = v3 - 24;
  if ( !v3 )
    v6 = 0;
  if ( v4 )
    v4 -= 24;
  return sub_134F310(a1, v4, v6, a3, 6u);
}
