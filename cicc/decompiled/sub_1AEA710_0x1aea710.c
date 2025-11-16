// Function: sub_1AEA710
// Address: 0x1aea710
//
__int64 __fastcall sub_1AEA710(__int64 a1, __int64 a2, _QWORD *a3, char a4, int a5, char a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdx

  v7 = *(_QWORD *)(a1 + 32);
  if ( v7 == *(_QWORD *)(a1 + 40) + 40LL || !v7 )
    v8 = 0;
  else
    v8 = v7 - 24;
  return sub_1AEA520(a1, a2, v8, a3, a4, a5, a6);
}
