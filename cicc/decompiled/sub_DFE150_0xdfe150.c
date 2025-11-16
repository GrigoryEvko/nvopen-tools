// Function: sub_DFE150
// Address: 0xdfe150
//
__int64 __fastcall sub_DFE150(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1480LL);
  if ( v1 == sub_DF62B0 )
    return 128;
  else
    return v1();
}
