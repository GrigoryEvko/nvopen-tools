// Function: sub_29BDD50
// Address: 0x29bdd50
//
__int64 __fastcall sub_29BDD50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdi

  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(a2 + 40);
  if ( v6 == v7 )
    return sub_B19DB0(a3, a1, a2);
  else
    return sub_29BD9C0(v7, v6, a3, a4, a1, a2);
}
