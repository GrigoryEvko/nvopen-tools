// Function: sub_3224A90
// Address: 0x3224a90
//
__int64 __fastcall sub_3224A90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // r15d
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // r13
  __int64 v8; // [rsp+8h] [rbp-48h]
  __int64 v9[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v3 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a1 + 8);
  v4 = sub_22077B0(0x310u);
  v5 = v4;
  if ( v4 )
    sub_37358C0(v4, v3, v2, v8, a1, a1 + 3776, 0);
  *(_QWORD *)(v5 + 56) = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 88);
  sub_3735CB0(v5);
  if ( *(_BYTE *)(a1 + 3770) )
    sub_324ACD0(v5);
  v9[0] = v5;
  sub_32249E0(a1, a2, v5 + 8, v9);
  v6 = v9[0];
  if ( v9[0] )
  {
    sub_3223CF0(v9[0]);
    j_j___libc_free_0(v6);
  }
  return v5;
}
