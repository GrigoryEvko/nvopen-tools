// Function: sub_C2F3D0
// Address: 0xc2f3d0
//
__int64 __fastcall sub_C2F3D0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 result; // rax
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_C2F370(*(_QWORD *)(a1 + 8));
  v2 = *(_QWORD *)(a1 + 8);
  v6[0] = 0;
  sub_CB6200(v2, v6, 8);
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 8);
  v6[0] = *(_QWORD *)(v3 + 120);
  sub_CB6200(v4, v6, 8);
  result = sub_F02A30(v3, v4);
  if ( *(_BYTE *)(a1 + 32) )
    return sub_C2EF90(*(_QWORD *)(a1 + 8), *(const void **)(a1 + 16), *(_QWORD *)(a1 + 24));
  return result;
}
