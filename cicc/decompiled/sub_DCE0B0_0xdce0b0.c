// Function: sub_DCE0B0
// Address: 0xdce0b0
//
unsigned __int64 __fastcall sub_DCE0B0(__int64 *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v11; // rax

  v4 = (_QWORD *)a2;
  v5 = sub_D95540(a2);
  v6 = sub_D97050((__int64)a1, v5);
  v7 = sub_D95540((__int64)a3);
  if ( v6 <= sub_D97050((__int64)a1, v7) )
  {
    v11 = sub_D95540((__int64)a3);
    v4 = sub_DC2CB0((__int64)a1, a2, v11);
  }
  else
  {
    v8 = sub_D95540(a2);
    a3 = sub_DC2B70((__int64)a1, (__int64)a3, v8, 0);
  }
  return sub_DCE050(a1, (__int64)v4, (__int64)a3, v9);
}
