// Function: sub_31E0DA0
// Address: 0x31e0da0
//
__int64 *__fastcall sub_31E0DA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // r8

  v3 = *(_QWORD *)(a1 + 448);
  if ( v3 )
    return sub_31E05D0(v3, a2);
  v5 = *(_QWORD *)(a1 + 216);
  v6 = sub_22077B0(0x60u);
  v3 = v6;
  if ( v6 )
  {
    *(_QWORD *)v6 = v5;
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)(v6 + 16) = 0;
    *(_QWORD *)(v6 + 24) = 0;
    *(_DWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    *(_QWORD *)(v6 + 56) = 0;
    *(_QWORD *)(v6 + 64) = 0;
    *(_QWORD *)(v6 + 72) = 0;
    *(_QWORD *)(v6 + 80) = 0;
    *(_DWORD *)(v6 + 88) = 0;
  }
  v7 = *(_QWORD *)(a1 + 448);
  *(_QWORD *)(a1 + 448) = v6;
  if ( !v7 )
    return sub_31E05D0(v3, a2);
  sub_31D8060(v7);
  return sub_31E05D0(*(_QWORD *)(a1 + 448), a2);
}
