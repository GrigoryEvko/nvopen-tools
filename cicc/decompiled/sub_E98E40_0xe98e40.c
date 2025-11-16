// Function: sub_E98E40
// Address: 0xe98e40
//
void __fastcall sub_E98E40(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 (*v5)(); // rcx
  __int64 v6; // rsi

  v2 = a1[3];
  for ( i = a1[4]; i != v2; *(_QWORD *)(v2 - 24) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v5)(a2, v6, a1[1]) )
  {
    while ( 1 )
    {
      if ( a2 )
      {
        v5 = *(__int64 (**)())(*(_QWORD *)a2 + 216LL);
        if ( v5 != sub_E974F0 )
          break;
      }
      *(_QWORD *)(v2 + 72) = 0;
      v2 += 96;
      if ( i == v2 )
        return;
    }
    v6 = v2;
    v2 += 96;
  }
}
