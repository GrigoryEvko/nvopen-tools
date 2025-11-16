// Function: sub_DFB730
// Address: 0xdfb730
//
__int64 __fastcall sub_DFB730(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1168LL);
  if ( v1 == sub_DF60F0 )
    return 1;
  else
    return v1();
}
