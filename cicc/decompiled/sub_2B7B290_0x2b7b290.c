// Function: sub_2B7B290
// Address: 0x2b7b290
//
__int64 ***__fastcall sub_2B7B290(__int64 *a1, __int64 a2, __int64 a3, int *a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD v13[4]; // [rsp+0h] [rbp-20h] BYREF

  v9 = a1[15];
  v10 = *(_QWORD *)(v9 + 3344);
  v13[0] = a1[14];
  v13[1] = v9 + 3112;
  v11 = *a1;
  v13[3] = v10;
  v13[2] = v9 + 3160;
  return sub_2B7A630(a2, a3, a4, a5, (__int64)v13, v11);
}
