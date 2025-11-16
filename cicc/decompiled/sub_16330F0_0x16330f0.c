// Function: sub_16330F0
// Address: 0x16330f0
//
_QWORD *__fastcall sub_16330F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rsi

  v2 = *(_QWORD *)(a2 + 168);
  if ( v2 )
    (*(void (**)(void))(*(_QWORD *)v2 + 32LL))();
  else
    *a1 = 1;
  return a1;
}
