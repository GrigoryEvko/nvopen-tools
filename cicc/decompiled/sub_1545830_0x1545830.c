// Function: sub_1545830
// Address: 0x1545830
//
void __fastcall sub_1545830(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // eax
  int v6; // r12d
  unsigned int i; // ebx
  __int64 v8; // rsi
  __int64 v9; // rax

  v5 = sub_161F520(a2, a2, a3, a4);
  if ( v5 )
  {
    v6 = v5;
    for ( i = 0; i != v6; ++i )
    {
      v8 = i;
      v9 = sub_161F530(a2, v8);
      sub_1545800(a1, 0, v9);
    }
  }
}
