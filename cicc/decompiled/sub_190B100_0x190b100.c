// Function: sub_190B100
// Address: 0x190b100
//
void __fastcall sub_190B100(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // r14
  _QWORD *v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r8d
  __int64 v11; // rsi
  int v12; // r10d
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned int v18; // eax

  v3 = *(_QWORD *)(a3 + 8);
  if ( v3 )
  {
    while ( 1 )
    {
      v6 = sub_1648700(v3);
      if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return;
    }
    v7 = (unsigned __int64)(unsigned int)(37 * a2) << 32;
    while ( 1 )
    {
      v8 = *(unsigned int *)(a1 + 176);
      v9 = v6[5];
      if ( (_DWORD)v8 )
      {
        v10 = v8 - 1;
        v11 = *(_QWORD *)(a1 + 160);
        v12 = 1;
        v13 = (((v7 | ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))
              - 1
              - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
            ^ ((v7 | ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))
             - 1
             - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
        v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
            ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
        v15 = (v8 - 1) & (((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - ((_DWORD)v14 << 27)));
        while ( 1 )
        {
          v16 = v11 + 24LL * v15;
          if ( a2 == *(_DWORD *)v16 && v9 == *(_QWORD *)(v16 + 8) )
            break;
          if ( *(_DWORD *)v16 == -1 )
          {
            if ( *(_QWORD *)(v16 + 8) == -8 )
              goto LABEL_12;
            v18 = v12 + v15;
            ++v12;
            v15 = v10 & v18;
          }
          else
          {
            v17 = v12 + v15;
            ++v12;
            v15 = v10 & v17;
          }
        }
        if ( v16 != v11 + 24 * v8 )
        {
          *(_DWORD *)v16 = -2;
          *(_QWORD *)(v16 + 8) = -16;
          --*(_DWORD *)(a1 + 168);
          ++*(_DWORD *)(a1 + 172);
        }
      }
LABEL_12:
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        break;
      while ( 1 )
      {
        v6 = sub_1648700(v3);
        if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return;
      }
    }
  }
}
