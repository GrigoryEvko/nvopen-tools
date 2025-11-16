// Function: sub_DFE210
// Address: 0xdfe210
//
__int64 __fastcall sub_DFE210(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1512LL);
  if ( v1 == sub_DF62C0 )
    return 1;
  else
    return v1();
}
