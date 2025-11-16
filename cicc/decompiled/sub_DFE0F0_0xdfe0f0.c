// Function: sub_DFE0F0
// Address: 0xdfe0f0
//
__int64 __fastcall sub_DFE0F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1464LL);
  if ( v1 == sub_DF62A0 )
    return 0;
  else
    return v1();
}
