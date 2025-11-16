// Function: sub_72F500
// Address: 0x72f500
//
__int64 __fastcall sub_72F500(__int64 a1, __int64 a2, char *a3, int a4, int a5)
{
  if ( a2 )
    return sub_72F3C0(*(_QWORD *)(a1 + 152), a2, a3, a4, a5);
  else
    return sub_72F3C0(*(_QWORD *)(a1 + 152), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), a3, a4, a5);
}
