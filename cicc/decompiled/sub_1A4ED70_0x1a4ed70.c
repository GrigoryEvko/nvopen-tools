// Function: sub_1A4ED70
// Address: 0x1a4ed70
//
__int64 *__fastcall sub_1A4ED70(__int64 a1, char a2, _QWORD **a3, __int64 a4)
{
  _QWORD **v5; // r13
  _QWORD **i; // rbx
  _QWORD *v8; // rsi
  __int64 *v9; // rdi
  _QWORD *v10; // rsi

  v5 = &a3[a4];
  for ( i = a3; v5 != i; ++i )
  {
    v8 = *i;
    sub_14070E0(*(__int64 **)(a1 + 8), v8);
  }
  v9 = *(__int64 **)(a1 + 8);
  v10 = **(_QWORD ***)a1;
  if ( a2 )
    return sub_14070E0(v9, v10);
  else
    return sub_1407870((__int64)v9, (__int64)v10);
}
