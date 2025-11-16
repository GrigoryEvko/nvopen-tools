// Function: sub_3015FE0
// Address: 0x3015fe0
//
void __fastcall sub_3015FE0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rsi

  if ( !*(_DWORD *)(a2 + 520) )
  {
    for ( i = *(_QWORD *)(a1 + 80); a1 + 72 != i; i = *(_QWORD *)(i + 8) )
    {
      v4 = 0;
      if ( i )
        v4 = i - 24;
      v5 = v4;
      v6 = sub_AA4FF0(v4);
      if ( !v6 )
        BUG();
      v7 = (unsigned int)*(unsigned __int8 *)(v6 - 24) - 39;
      if ( (unsigned int)v7 <= 0x38 )
      {
        v8 = 0x100060000000001LL;
        if ( _bittest64(&v8, v7) )
        {
          v10 = sub_AA4FF0(v5);
          if ( v10 )
            v10 -= 24;
          if ( sub_3011FC0(v10) )
            sub_3014680(a2, v10, 0xFFFFFFFF);
        }
      }
    }
    sub_30138B0(a1, a2);
    if ( sub_BA91D0(*(_QWORD *)(a1 + 40), "eh-asynch", 9u) )
    {
      v9 = *(_QWORD *)(a1 + 80);
      if ( v9 )
        v9 -= 24;
      sub_30158E0(v9, -1, a2);
    }
  }
}
