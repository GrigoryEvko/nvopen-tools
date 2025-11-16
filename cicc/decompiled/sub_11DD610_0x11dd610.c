// Function: sub_11DD610
// Address: 0x11dd610
//
__int64 __fastcall sub_11DD610(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // r12
  _BYTE *v10; // rax
  unsigned int v12[14]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v3));
  *(_QWORD *)v12 = 0x100000000LL;
  sub_11DA4B0(a2, (int *)v12, 2);
  v6 = sub_98B430(v5, 8u);
  if ( !v6 )
    return 0;
  v7 = v6;
  v12[0] = 1;
  sub_11DA2E0(a2, v12, 1, v6);
  v8 = v7 - 1;
  if ( !v8 )
    return v4;
  v10 = sub_11DD500(a1, v5, v4, v8, a3);
  v4 = (__int64)v10;
  if ( !v10 )
    return 0;
  if ( *v10 == 85 )
    *((_WORD *)v10 + 1) = *((_WORD *)v10 + 1) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return v4;
}
