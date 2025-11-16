// Function: sub_DFE1E0
// Address: 0xdfe1e0
//
__int64 __fastcall sub_DFE1E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1504LL);
  if ( v1 == sub_DF62C0 )
    return 1;
  else
    return v1();
}
