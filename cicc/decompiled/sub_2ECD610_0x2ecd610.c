// Function: sub_2ECD610
// Address: 0x2ecd610
//
unsigned __int64 __fastcall sub_2ECD610(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax

  v1 = a1 + 4480;
  if ( *(_BYTE *)(a1 + 4016) )
  {
    sub_2F796A0(
      a1 + 4480,
      *(_QWORD *)(a1 + 32),
      *(_QWORD *)(a1 + 3544),
      *(_QWORD *)(a1 + 3464),
      *(_QWORD *)(a1 + 904),
      *(_QWORD *)(a1 + 3632),
      *(unsigned __int8 *)(a1 + 4017),
      1);
    if ( *(_QWORD *)(a1 + 3632) != *(_QWORD *)(a1 + 920) )
      sub_2F78B80(v1, 0);
    sub_2F97F60(a1, *(_QWORD *)(a1 + 3456), v1, a1 + 4000, *(_QWORD *)(a1 + 3464), *(unsigned __int8 *)(a1 + 4017));
    return sub_2ECD150(a1);
  }
  else
  {
    sub_2F75310(a1 + 4480);
    v3 = *(_QWORD *)(a1 + 4896);
    if ( v3 != *(_QWORD *)(a1 + 4904) )
      *(_QWORD *)(a1 + 4904) = v3;
    return sub_2F97F60(a1, *(_QWORD *)(a1 + 3456), 0, 0, 0, 0);
  }
}
