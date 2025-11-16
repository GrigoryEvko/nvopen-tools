// Function: sub_31F4E40
// Address: 0x31f4e40
//
__int64 __fastcall sub_31F4E40(__int64 a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(void); // rcx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(__int64 (**)(void))(*(_QWORD *)v1 + 552LL);
  if ( (char *)v2 == (char *)sub_C13FB0 )
    return (*(__int64 (**)(void))(*(_QWORD *)v1 + 536LL))();
  else
    return v2();
}
