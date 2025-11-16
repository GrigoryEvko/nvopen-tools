// Function: sub_14A2F90
// Address: 0x14a2f90
//
__int64 __fastcall sub_14A2F90(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 408LL);
  if ( v1 == sub_14A0920 )
    return 0;
  else
    return v1();
}
