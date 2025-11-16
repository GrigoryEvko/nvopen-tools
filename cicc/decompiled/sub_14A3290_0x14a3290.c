// Function: sub_14A3290
// Address: 0x14a3290
//
__int64 __fastcall sub_14A3290(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 552LL);
  if ( v1 == sub_14A08D0 )
    return 0;
  else
    return v1();
}
