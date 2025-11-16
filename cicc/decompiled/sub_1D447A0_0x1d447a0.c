// Function: sub_1D447A0
// Address: 0x1d447a0
//
__int64 __fastcall sub_1D447A0(
        const __m128i *a1,
        __int64 a2,
        __int16 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9

  v8 = sub_1D2E3C0((__int64)a1, a2, ~a3, a4, a5, a6, a7, a8);
  *(_DWORD *)(v8 + 28) = -1;
  v9 = v8;
  if ( a2 != v8 )
  {
    sub_1D444E0((__int64)a1, a2, v8);
    sub_1D2DC70(a1, a2, v10, v11, v12, v13);
  }
  return v9;
}
