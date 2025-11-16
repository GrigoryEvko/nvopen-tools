// Function: sub_B833D0
// Address: 0xb833d0
//
unsigned __int64 __fastcall sub_B833D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v5; // r13d
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(unsigned int *)(a3 + 8);
  v5 = *(unsigned __int8 *)(a2 + 168);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v4 + 1, 4);
    v4 = *(unsigned int *)(a3 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a3 + 4 * v4) = v5;
  ++*(_DWORD *)(a3 + 8);
  v7[0] = a3;
  sub_B803F0(v7, a2 + 8);
  sub_B803F0(v7, a2 + 88);
  sub_B803F0(v7, a2 + 120);
  sub_B803F0(v7, a2 + 152);
  return sub_939680(*(_QWORD **)a3, *(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
}
