// Function: sub_1594E40
// Address: 0x1594e40
//
__int64 __fastcall sub_1594E40(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rcx
  int v7; // edx
  int v8; // r8d
  unsigned int v9; // edi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rsi
  unsigned int i; // eax
  _QWORD *v13; // rsi
  unsigned int v14; // eax

  v1 = 0;
  if ( *(_WORD *)(a1 + 18) )
  {
    v2 = *(_QWORD *)(a1 + 56);
    v4 = *(_QWORD *)sub_15E0530(v2);
    v5 = *(_DWORD *)(v4 + 1768);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v4 + 1752);
      v7 = v5 - 1;
      v8 = 1;
      v9 = (unsigned int)a1 >> 9;
      v10 = (((v9 ^ ((unsigned int)a1 >> 4)
             | ((unsigned __int64)(((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v9 ^ ((unsigned int)a1 >> 4)) << 32)) >> 22)
          ^ ((v9 ^ ((unsigned int)a1 >> 4)
            | ((unsigned __int64)(((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v9 ^ ((unsigned int)a1 >> 4)) << 32));
      v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
          ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
      for ( i = v7 & (((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27))); ; i = v7 & v14 )
      {
        v13 = (_QWORD *)(v6 + 24LL * i);
        if ( v2 == *v13 && a1 == v13[1] )
          return v13[2];
        if ( *v13 == -8 && v13[1] == -8 )
          break;
        v14 = v8 + i;
        ++v8;
      }
      return 0;
    }
  }
  return v1;
}
