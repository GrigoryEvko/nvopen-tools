// Function: sub_2545E30
// Address: 0x2545e30
//
void __fastcall sub_2545E30(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 v3; // r9
  __int64 v4; // rsi
  int v5; // ecx
  __int64 v6; // r8
  int v7; // r12d
  unsigned int i; // eax
  _QWORD *v9; // rdx
  unsigned int v10; // eax

  v2 = *(_DWORD *)(a2 + 56);
  v3 = *(_QWORD *)(a2 + 40);
  if ( v2 )
  {
    v4 = *(_QWORD *)(a1 + 72);
    v5 = v2 - 1;
    v6 = *(_QWORD *)(a1 + 80);
    v7 = 1;
    for ( i = v5
            & (((unsigned int)v6 >> 9)
             ^ ((unsigned int)v6 >> 4)
             ^ (16 * (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)))); ; i = v5 & v10 )
    {
      v9 = (_QWORD *)(v3 + ((unsigned __int64)i << 6));
      if ( v4 == *v9 && v6 == v9[1] )
        break;
      if ( unk_4FEE4D0 == *v9 && unk_4FEE4D8 == v9[1] )
        return;
      v10 = v7 + i;
      ++v7;
    }
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
}
