// Function: sub_B89400
// Address: 0xb89400
//
__int64 __fastcall sub_B89400(__int64 *a1, __int64 a2, int a3)
{
  int v4; // eax

  while ( 1 )
  {
    v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a2 + 8) - 8LL) + 40LL))(*(_QWORD *)(*(_QWORD *)(a2 + 8) - 8LL));
    if ( v4 <= 1 || v4 == a3 )
      break;
    sub_B823C0(a2);
  }
  return sub_B88F40(*(_QWORD *)(*(_QWORD *)(a2 + 8) - 8LL), a1, 1);
}
