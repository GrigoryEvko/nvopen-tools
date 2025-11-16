// Function: sub_315E1E0
// Address: 0x315e1e0
//
int __fastcall sub_315E1E0(__int64 a1, const char **a2)
{
  __int64 v2; // r13
  __int64 v3; // rax

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(v2 + 56);
  if ( v3 && a1 == v3 - 24 && sub_AA54C0(*(_QWORD *)(a1 + 40)) )
    return sub_BD6B50((unsigned __int8 *)v2, a2);
  else
    return sub_AA8550((_QWORD *)v2, (__int64 *)(a1 + 24), 0, (__int64)a2, 0);
}
