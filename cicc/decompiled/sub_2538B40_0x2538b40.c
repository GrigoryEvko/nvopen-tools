// Function: sub_2538B40
// Address: 0x2538b40
//
__int64 __fastcall sub_2538B40(_QWORD *a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*a1 + 112LL);
  if ( (char *)v1 == (char *)sub_2534E30 )
    return a1[19];
  else
    return *(_QWORD *)(v1() + 32);
}
