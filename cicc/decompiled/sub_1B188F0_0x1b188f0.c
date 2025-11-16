// Function: sub_1B188F0
// Address: 0x1b188f0
//
void __fastcall sub_1B188F0(unsigned __int8 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rbx
  unsigned __int8 v6; // r15
  __int64 v7; // rsi
  __int64 *v8; // r14
  __int64 v9; // rsi
  unsigned __int8 v10; // al

  if ( a1[16] > 0x17u )
  {
    v5 = a2;
    if ( a4 )
    {
      v6 = *(_BYTE *)(a4 + 16);
      v7 = a4;
      if ( v6 <= 0x17u )
        return;
    }
    else
    {
      v7 = *a2;
      v6 = *(_BYTE *)(v7 + 16);
      if ( v6 <= 0x17u )
        return;
    }
    v8 = &v5[a3];
    sub_15F2530(a1, v7, 1);
    while ( v8 != v5 )
    {
      while ( 1 )
      {
        v9 = *v5;
        v10 = *(_BYTE *)(*v5 + 16);
        if ( v10 > 0x17u && (v6 == v10 || !a4) )
          break;
        if ( v8 == ++v5 )
          return;
      }
      ++v5;
      sub_15F2780(a1, v9);
    }
  }
}
