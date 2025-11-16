// Function: sub_255C130
// Address: 0x255c130
//
__int64 __fastcall sub_255C130(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // edx
  int v6; // r14d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // r13
  unsigned int i; // eax
  __int64 *v12; // r12
  __int64 v13; // r15
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = 1;
    v7 = *(_QWORD *)(a1 + 8);
    v8 = a2[1];
    v9 = *a2;
    v10 = 0;
    for ( i = v5
            & (((unsigned int)v8 >> 9)
             ^ ((unsigned int)v8 >> 4)
             ^ (16 * (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9)))); ; i = v5 & v14 )
    {
      v12 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
      v13 = *v12;
      if ( v9 == *v12 && v8 == v12[1] )
      {
        *a3 = v12;
        return 1;
      }
      if ( v13 == unk_4FEE4D0 && unk_4FEE4D8 == v12[1] )
        break;
      if ( v13 == qword_4FEE4C0[0] && v12[1] == qword_4FEE4C0[1] && !v10 )
        v10 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
      v14 = v6 + i;
      ++v6;
    }
    if ( !v10 )
      v10 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
    *a3 = v10;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
