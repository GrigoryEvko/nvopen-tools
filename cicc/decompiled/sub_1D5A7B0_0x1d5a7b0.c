// Function: sub_1D5A7B0
// Address: 0x1d5a7b0
//
__int64 __fastcall sub_1D5A7B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 576LL);
  if ( v1 == sub_1D12D90 )
    return 0;
  else
    return v1();
}
