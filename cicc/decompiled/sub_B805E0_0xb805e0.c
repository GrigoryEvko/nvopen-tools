// Function: sub_B805E0
// Address: 0xb805e0
//
__int64 __fastcall sub_B805E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  int v8; // r13d
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = *(unsigned int *)(a5 + 8);
  v8 = *(unsigned __int8 *)(a2 + 168);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
  {
    sub_C8D5F0(a5, a5 + 16, v7 + 1, 4);
    v7 = *(unsigned int *)(a5 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a5 + 4 * v7) = v8;
  ++*(_DWORD *)(a5 + 8);
  v10[0] = a5;
  sub_B803F0(v10, a2 + 8);
  sub_B803F0(v10, a2 + 88);
  sub_B803F0(v10, a2 + 120);
  sub_B803F0(v10, a2 + 152);
  return sub_C656C0(a5, a3);
}
