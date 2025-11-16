// Function: sub_16FF990
// Address: 0x16ff990
//
_QWORD *__fastcall sub_16FF990(_QWORD *a1, __int64 a2, __int64 a3)
{
  void (*v3)(void); // rax
  __int64 v4; // rax

  v3 = *(void (**)(void))(**(_QWORD **)a2 + 40LL);
  if ( (char *)v3 == (char *)sub_16FF7D0 )
  {
    v4 = sub_1632FA0(*(_QWORD *)(a3 + 40));
    sub_14A2630(a1, v4);
  }
  else
  {
    v3();
  }
  return a1;
}
