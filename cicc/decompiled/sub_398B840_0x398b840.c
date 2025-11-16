// Function: sub_398B840
// Address: 0x398b840
//
void __fastcall sub_398B840(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_39A3F30(
    *a4,
    a3,
    8496,
    *(_QWORD *)(*(_QWORD *)(a1[1] + 232LL) + 880LL),
    *(_QWORD *)(*(_QWORD *)(a1[1] + 232LL) + 888LL));
  if ( a1[504] )
    sub_39A3F30(*a4, a3, 27, a1[503], a1[504]);
  sub_3989C90((__int64)a1, *a4, a3);
  v6 = *a4;
  *a4 = 0;
  v7[0] = v6;
  sub_39A0610(a1 + 565, v7);
  if ( v7[0] )
    sub_3985790(v7[0]);
}
