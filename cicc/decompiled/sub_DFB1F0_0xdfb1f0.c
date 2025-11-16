// Function: sub_DFB1F0
// Address: 0xdfb1f0
//
__int64 __fastcall sub_DFB1F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1016LL);
  if ( v1 == sub_DF6030 )
    return 128;
  else
    return v1();
}
