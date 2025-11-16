// Function: sub_1AA6530
// Address: 0x1aa6530
//
__int64 __fastcall sub_1AA6530(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 *v10; // rax
  bool v11; // zf
  __int64 v12; // rdi
  unsigned __int64 *v14; // [rsp+8h] [rbp-8h] BYREF

  v10 = (unsigned __int64 *)(a1 + 24);
  v11 = a1 == 0;
  v12 = *(_QWORD *)(a1 + 40);
  if ( v11 )
    v10 = 0;
  v14 = v10;
  return sub_1AA6440(v12 + 40, &v14, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
