// Function: sub_3415C10
// Address: 0x3415c10
//
__int64 __fastcall sub_3415C10(
        const __m128i *a1,
        __int64 a2,
        int a3,
        unsigned __int64 a4,
        int a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v13; // [rsp-8h] [rbp-28h]

  v8 = sub_33EC480((__int64)a1, a2, ~a3, a4, a5, a6, a7, a8);
  *(_DWORD *)(v8 + 36) = -1;
  v11 = v8;
  if ( a2 != v8 )
  {
    sub_34158F0((__int64)a1, a2, v8, v13, v9, v10);
    sub_33ECEA0(a1, a2);
  }
  return v11;
}
