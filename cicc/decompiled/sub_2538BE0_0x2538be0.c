// Function: sub_2538BE0
// Address: 0x2538be0
//
__int64 __fastcall sub_2538BE0(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v2; // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 112LL);
  if ( (char *)v1 == (char *)sub_2534E30 )
    v2 = a1 + 120;
  else
    v2 = v1();
  return *(_QWORD *)(v2 + 32) + 8LL * *(unsigned int *)(v2 + 40);
}
