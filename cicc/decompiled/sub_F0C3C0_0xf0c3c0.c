// Function: sub_F0C3C0
// Address: 0xf0c3c0
//
__int64 __fastcall sub_F0C3C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(***(_QWORD ***)(a1 + 8) + 200LL);
  if ( v1 == sub_DF5C80 )
    return 1;
  else
    return v1();
}
