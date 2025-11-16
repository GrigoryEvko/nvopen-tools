// Function: sub_14A35C0
// Address: 0x14a35c0
//
__int64 __fastcall sub_14A35C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 704LL);
  if ( v1 == sub_14A0AB0 )
    return 1;
  else
    return v1();
}
