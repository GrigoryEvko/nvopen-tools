// Function: sub_DF94A0
// Address: 0xdf94a0
//
__int64 __fastcall sub_DF94A0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 88LL);
  if ( v1 == sub_DF5BD0 )
    return 0;
  else
    return v1();
}
