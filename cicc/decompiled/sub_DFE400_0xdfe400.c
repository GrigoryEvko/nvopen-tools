// Function: sub_DFE400
// Address: 0xdfe400
//
__int64 __fastcall sub_DFE400(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1664LL);
  if ( v1 == sub_DF6310 )
    return 0x200000001LL;
  else
    return v1();
}
