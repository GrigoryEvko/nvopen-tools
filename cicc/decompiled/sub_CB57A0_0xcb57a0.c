// Function: sub_CB57A0
// Address: 0xcb57a0
//
__int64 __fastcall sub_CB57A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax

  v2 = a1[6];
  v3 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL))(a1);
  return sub_2240E30(v2, v3 + a2 + a1[4] - a1[2]);
}
