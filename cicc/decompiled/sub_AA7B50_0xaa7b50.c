// Function: sub_AA7B50
// Address: 0xaa7b50
//
void __fastcall sub_AA7B50(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r15
  __int64 v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdi

  if ( *(_BYTE *)(a1 + 40) )
  {
    v6 = a4 + 48;
    if ( a4 + 48 == (*(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( sub_AA60B0(a4) )
      {
        v11 = a2 - 24;
        if ( !a2 )
          v11 = 0;
        sub_B44050(v11, a4, v6, 0, a3);
      }
    }
    else if ( *(_QWORD *)(a4 + 56) == a5 && a6 )
    {
      v9 = a5 - 24;
      if ( !a5 )
        v9 = 0;
      if ( (unsigned __int8)sub_B44020(v9) )
      {
        v10 = sub_AA7AD0(a1, a2);
        sub_B14410(v10, *(_QWORD *)(v9 + 64), a3);
      }
    }
  }
}
