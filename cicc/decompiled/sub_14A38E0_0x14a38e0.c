// Function: sub_14A38E0
// Address: 0x14a38e0
//
__int64 __fastcall sub_14A38E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 808LL);
  if ( v1 == sub_14A08C0 )
    return 1;
  else
    return v1();
}
