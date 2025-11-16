// Function: sub_DF94D0
// Address: 0xdf94d0
//
__int64 __fastcall sub_DF94D0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 80LL);
  if ( v1 == sub_DF5BC0 )
    return 150;
  else
    return v1();
}
