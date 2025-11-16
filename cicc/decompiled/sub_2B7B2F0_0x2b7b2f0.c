// Function: sub_2B7B2F0
// Address: 0x2b7b2f0
//
__int64 ***__fastcall sub_2B7B2F0(__int64 **a1, __int64 a2, __int64 a3, int *a4, __int64 a5)
{
  __int64 *v10; // r8
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r9
  _QWORD v15[4]; // [rsp+0h] [rbp-20h] BYREF

  v10 = *a1;
  v11 = (*a1)[15];
  v12 = (*a1)[14];
  v13 = *(_QWORD *)(v11 + 3344);
  v15[0] = v12;
  v15[1] = v11 + 3112;
  v15[2] = v11 + 3160;
  v15[3] = v13;
  return sub_2B7A630(a2, a3, a4, a5, (__int64)v15, *v10);
}
