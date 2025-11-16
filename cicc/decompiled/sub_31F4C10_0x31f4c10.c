// Function: sub_31F4C10
// Address: 0x31f4c10
//
__int64 __fastcall sub_31F4C10(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 8) + 96LL);
  if ( v1 == sub_C13EE0 )
    return 0;
  else
    return v1();
}
