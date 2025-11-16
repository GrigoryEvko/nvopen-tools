// Function: sub_FA3020
// Address: 0xfa3020
//
__int64 __fastcall sub_FA3020(__int64 a1, __int64 a2, __int64 a3, unsigned int **a4)
{
  unsigned __int64 v7; // rsi
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rbx

  v7 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == a3 + 48 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = v7 - 24;
    if ( (unsigned int)(v8 - 30) >= 0xB )
      v9 = 0;
  }
  v10 = sub_F8E5E0(a1, v9);
  if ( v10 && v10 == sub_F8E5E0(a1, a2) )
    return sub_FA2310(a1, a2, a3, a4);
  else
    return 0;
}
