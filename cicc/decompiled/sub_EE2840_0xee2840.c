// Function: sub_EE2840
// Address: 0xee2840
//
unsigned __int64 *__fastcall sub_EE2840(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  _QWORD *v8; // rax
  size_t *v9; // rdx
  size_t *v10; // rcx
  size_t *v12[9]; // [rsp+0h] [rbp-50h] BYREF

  v8 = *(_QWORD **)(a2 + 8);
  v9 = (size_t *)v8[1];
  v10 = (size_t *)v8[9];
  v12[1] = 0;
  v12[3] = v8 + 4;
  v12[2] = v9;
  v12[0] = v10;
  memset(&v12[4], 0, 32);
  sub_EE2740(a1, a3, v12, (__int64)v10, a2, a6);
  return a1;
}
