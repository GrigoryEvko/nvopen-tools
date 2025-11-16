// Function: sub_BA9820
// Address: 0xba9820
//
_QWORD *__fastcall sub_BA9820(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rsi

  v2 = *(_QWORD *)(a2 + 160);
  if ( v2 )
    (*(void (**)(void))(*(_QWORD *)v2 + 32LL))();
  else
    *a1 = 1;
  return a1;
}
