// Function: sub_335DD50
// Address: 0x335dd50
//
__int64 __fastcall sub_335DD50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax

  a1[73] = a3;
  a1[74] = a2;
  sub_2F8EBD0((__int64)a1, a2, a3, a4, a5, a6);
  v6 = a1[76];
  if ( v6 != a1[77] )
    a1[77] = v6;
  return (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 88LL))(a1);
}
