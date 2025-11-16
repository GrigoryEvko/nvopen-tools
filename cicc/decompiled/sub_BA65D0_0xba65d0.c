// Function: sub_BA65D0
// Address: 0xba65d0
//
__int64 __fastcall sub_BA65D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8

  v6 = *(_QWORD *)(a1 + 8);
  if ( (v6 & 4) == 0 )
    return sub_B97380(a1, a2, a3, a4, a5);
  sub_BA6110((const __m128i *)(v6 & 0xFFFFFFFFFFFFFFF8LL), 0);
  return sub_B97380(a1, 0, v8, v9, v10);
}
