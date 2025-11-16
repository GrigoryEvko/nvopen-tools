// Function: sub_CB57E0
// Address: 0xcb57e0
//
__int64 __fastcall sub_CB57E0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rdx

  v2 = a1[6];
  result = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL))(a1);
  v6 = result + a2 + a1[4] - a1[2];
  if ( v6 > *(_QWORD *)(v2 + 16) )
    return sub_C8D290(v2, (const void *)(v2 + 24), v6, 1u, v4, v5);
  return result;
}
