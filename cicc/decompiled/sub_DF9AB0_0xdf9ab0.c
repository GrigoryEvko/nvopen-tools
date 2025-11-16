// Function: sub_DF9AB0
// Address: 0xdf9ab0
//
__int64 __fastcall sub_DF9AB0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 280LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
