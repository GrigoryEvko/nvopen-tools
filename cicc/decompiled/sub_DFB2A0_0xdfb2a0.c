// Function: sub_DFB2A0
// Address: 0xdfb2a0
//
__int64 __fastcall sub_DFB2A0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1040LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
