// Function: sub_DFDBE0
// Address: 0xdfdbe0
//
__int64 __fastcall sub_DFDBE0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 104LL);
  if ( v1 == sub_DF5BF0 )
    return 64;
  else
    return v1();
}
