// Function: sub_31DA6B0
// Address: 0x31da6b0
//
__int64 __fastcall sub_31DA6B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 200) + 24LL);
  if ( v1 == sub_23CE280 )
    return 0;
  else
    return v1();
}
