// Function: sub_91FFA0
// Address: 0x91ffa0
//
__int64 __fastcall sub_91FFA0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v6[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = *(_QWORD *)(a1 + 344);
  v6[0] = a1;
  v6[1] = 0;
  v6[2] = 0;
  v6[3] = v4;
  return sub_91DF90(v6, a2, a3, a4);
}
