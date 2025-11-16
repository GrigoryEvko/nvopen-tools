// Function: sub_1E77580
// Address: 0x1e77580
//
__int64 __fastcall sub_1E77580(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 v4; // [rsp-8h] [rbp-18h]

  v1 = a1 + 2776;
  if ( *(_BYTE *)(a1 + 2568) )
  {
    sub_1EE96B0(
      a1 + 2776,
      *(_QWORD *)(a1 + 32),
      *(_QWORD *)(a1 + 2272),
      *(_QWORD *)(a1 + 2112),
      *(_QWORD *)(a1 + 920),
      *(_QWORD *)(a1 + 2312),
      *(unsigned __int8 *)(a1 + 2569),
      1);
    if ( *(_QWORD *)(a1 + 2312) != *(_QWORD *)(a1 + 936) )
      sub_1EE8D00(v1, 0, v4);
    sub_1F0A020(a1, *(_QWORD *)(a1 + 2104), v1, a1 + 2552, *(_QWORD *)(a1 + 2112), *(unsigned __int8 *)(a1 + 2569));
    return sub_1E770E0(a1);
  }
  else
  {
    sub_1EE6140(a1 + 2776);
    v3 = *(_QWORD *)(a1 + 3064);
    if ( v3 != *(_QWORD *)(a1 + 3072) )
      *(_QWORD *)(a1 + 3072) = v3;
    return sub_1F0A020(a1, *(_QWORD *)(a1 + 2104), 0, 0, 0, 0);
  }
}
