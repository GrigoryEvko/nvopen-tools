// Function: sub_DFDDB0
// Address: 0xdfddb0
//
__int64 __fastcall sub_DFDDB0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1416LL);
  if ( v1 == sub_DF6280 )
    return 0;
  else
    return v1();
}
