// Function: sub_2FDBDE0
// Address: 0x2fdbde0
//
__int64 __fastcall sub_2FDBDE0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 48LL);
  if ( v1 == sub_2FDBB70 )
    return 0;
  else
    return v1();
}
