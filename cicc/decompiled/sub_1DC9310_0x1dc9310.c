// Function: sub_1DC9310
// Address: 0x1dc9310
//
__int64 __fastcall sub_1DC9310(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rdx

  v2 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v2 == sub_1D00B10 )
    *(_QWORD *)(a1 + 232) = 0;
  else
    *(_QWORD *)(a1 + 232) = v2();
  return 0;
}
