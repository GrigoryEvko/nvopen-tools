// Function: sub_14A3350
// Address: 0x14a3350
//
__int64 __fastcall sub_14A3350(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 584LL);
  if ( v1 == sub_14A09D0 )
    return 1;
  else
    return v1();
}
