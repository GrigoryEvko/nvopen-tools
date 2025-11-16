// Function: sub_26E2030
// Address: 0x26e2030
//
bool __fastcall sub_26E2030(__int64 a1, __int64 a2)
{
  int *v2; // r14
  size_t v3; // r13
  _QWORD *v4; // r15
  size_t v6[2]; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v7[24]; // [rsp+10h] [rbp-C0h] BYREF

  v2 = *(int **)a2;
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD **)(a1 + 264);
  if ( *(_QWORD *)a2 )
  {
    sub_C7D030(v7);
    sub_C7D280((int *)v7, v2, v3);
    sub_C7D290(v7, v6);
    v3 = v6[0];
  }
  v7[0] = v3;
  return sub_26C56D0(v4, v7) == 0;
}
