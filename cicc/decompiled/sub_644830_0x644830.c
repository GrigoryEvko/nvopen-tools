// Function: sub_644830
// Address: 0x644830
//
void __fastcall sub_644830(__int64 a1, char a2, __int64 a3, int a4)
{
  __int64 *v7; // rdi

  if ( *(_QWORD *)(a1 + 200) || *(_QWORD *)(a1 + 184) )
  {
    if ( *(char *)(a1 + 121) < 0 )
      *(_QWORD *)(a1 + 184) = sub_5CF190(*(const __m128i **)(a1 + 184));
    sub_6447A0(a1);
    v7 = *(__int64 **)(a1 + 200);
    if ( a4 )
    {
      sub_5CF700(v7);
      sub_5CEC90(*(_QWORD **)(a1 + 200), a3, a2);
      sub_5CF700(*(__int64 **)(a1 + 184));
    }
    else
    {
      sub_5CEC90(v7, a3, a2);
    }
    sub_5CEC90(*(_QWORD **)(a1 + 184), a3, a2);
    sub_6447E0(a1);
  }
}
