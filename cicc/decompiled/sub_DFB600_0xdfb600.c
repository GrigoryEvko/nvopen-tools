// Function: sub_DFB600
// Address: 0xdfb600
//
__int64 __fastcall sub_DFB600(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1120LL);
  if ( v1 == sub_DF60A0 )
    return 0;
  else
    return v1();
}
