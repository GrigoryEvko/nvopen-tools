// Function: sub_DFE5E0
// Address: 0xdfe5e0
//
__int64 __fastcall sub_DFE5E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1616LL);
  if ( v1 == sub_DF60A0 )
    return 0;
  else
    return v1();
}
