// Function: sub_14A3260
// Address: 0x14a3260
//
__int64 __fastcall sub_14A3260(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 528LL);
  if ( v1 == sub_14A08D0 )
    return 0;
  else
    return v1();
}
