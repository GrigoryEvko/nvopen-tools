// Function: sub_C60570
// Address: 0xc60570
//
__int64 __fastcall sub_C60570(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  _BYTE *v5; // rax
  __int64 v7; // rax

  v4 = *a1;
  if ( *a1 != a1[1] )
  {
    a2 = sub_CB59F0(a2, v4);
    v5 = *(_BYTE **)(a2 + 32);
    if ( *(_BYTE **)(a2 + 24) == v5 )
    {
      v7 = sub_CB6200(a2, "-", 1);
      return sub_CB59F0(v7, a1[1]);
    }
    *v5 = 45;
    ++*(_QWORD *)(a2 + 32);
    v4 = a1[1];
  }
  return sub_CB59F0(a2, v4);
}
