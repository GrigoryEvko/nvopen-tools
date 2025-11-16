// Function: sub_91FFE0
// Address: 0x91ffe0
//
__int64 __fastcall sub_91FFE0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v6[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = *(_QWORD *)(a1 + 32);
  v6[1] = a1;
  v6[0] = v4;
  v6[2] = a1 + 48;
  v6[3] = *(_QWORD *)(a1 + 40);
  return sub_91DF90(v6, a2, a3, a4);
}
