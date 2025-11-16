// Function: sub_2FE3CB0
// Address: 0x2fe3cb0
//
__int64 __fastcall sub_2FE3CB0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 384LL);
  if ( v1 == sub_2FE3020 )
    return 0;
  else
    return v1();
}
