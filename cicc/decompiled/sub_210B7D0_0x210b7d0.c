// Function: sub_210B7D0
// Address: 0x210b7d0
//
void __fastcall sub_210B7D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  int v6; // esi
  __int64 v7; // rax
  int v8; // eax
  int v9; // eax

  v3 = *(_QWORD *)(a2 + 16);
  if ( **(_WORD **)(v3 + 16) && **(_WORD **)(v3 + 16) != 45 )
  {
    v9 = sub_210AD80(a1, *(_QWORD **)(v3 + 24));
    sub_1E310D0(a2, v9);
  }
  else
  {
    v4 = *(_QWORD *)(v3 + 32);
    if ( a2 == v4 + 40 )
    {
      v7 = 80;
    }
    else
    {
      LODWORD(v5) = 1;
      do
      {
        v6 = v5;
        v5 = (unsigned int)(v5 + 2);
      }
      while ( a2 != v4 + 40 * v5 );
      v7 = 40LL * (unsigned int)(v6 + 3);
    }
    v8 = sub_210AA20(a1, *(_QWORD *)(v4 + v7 + 24));
    sub_1E310D0(a2, v8);
  }
}
