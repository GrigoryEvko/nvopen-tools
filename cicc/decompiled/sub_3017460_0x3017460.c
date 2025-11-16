// Function: sub_3017460
// Address: 0x3017460
//
void __fastcall sub_3017460(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdi
  const __m128i *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9

  if ( !*(_DWORD *)(a2 + 16) )
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
          v10 = (const __m128i *)sub_AA4FF0(v5);
          if ( v10 )
            v10 = (const __m128i *)((char *)v10 - 24);
          if ( sub_3011FC0((__int64)v10) )
            sub_3016BA0(a2, v10, 0xFFFFFFFF, v11, v12, v13);
        }
      }
    }
    sub_30138B0(a1, a2);
    if ( sub_BA91D0(*(_QWORD *)(a1 + 40), "eh-asynch", 9u) )
    {
      v9 = *(_QWORD *)(a1 + 80);
      if ( v9 )
        v9 -= 24;
      sub_3016110(v9, -1, a2);
    }
  }
}
