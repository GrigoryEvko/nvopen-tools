// Function: sub_396DD80
// Address: 0x396dd80
//
__int64 __fastcall sub_396DD80(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 232) + 24LL);
  if ( v1 == sub_16FF760 )
    return 0;
  else
    return v1();
}
