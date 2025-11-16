// Function: sub_22119A0
// Address: 0x22119a0
//
__int64 __fastcall sub_22119A0(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 *a6)
{
  __int64 v7; // r9
  __int64 v8; // rdi
  char v10; // [rsp+17h] [rbp-41h] BYREF
  _QWORD v11[4]; // [rsp+18h] [rbp-40h] BYREF
  void (__fastcall *v12)(_QWORD *); // [rsp+38h] [rbp-20h]

  v7 = *a6;
  v8 = *(_QWORD *)(a2 + 32);
  v12 = 0;
  sub_2222300(v8, (unsigned int)v11, a3, a4, a5, v7, *(_QWORD *)(v7 - 24));
  if ( !v12 )
    sub_426248((__int64)"uninitialized __any_string");
  sub_2216C30(a1, v11[0], v11[1], &v10);
  if ( v12 )
    v12(v11);
  return a1;
}
