// Function: sub_AC3FA0
// Address: 0xac3fa0
//
__int64 __fastcall sub_AC3FA0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 v3; // rax
  int v4; // esi
  __int64 v5; // rcx
  int v6; // esi
  int v7; // r8d
  unsigned int i; // eax
  _QWORD *v9; // rdx
  unsigned int v10; // eax

  v1 = 0;
  if ( (*(_WORD *)(a1 + 2) & 0x7FFF) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 72);
    v3 = *(_QWORD *)sub_B2BE50(v2);
    v4 = *(_DWORD *)(v3 + 2016);
    v5 = *(_QWORD *)(v3 + 2000);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = 1;
      for ( i = v6
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)
                  | ((unsigned __int64)(((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)))); ; i = v6 & v10 )
      {
        v9 = (_QWORD *)(v5 + 24LL * i);
        if ( v2 == *v9 && a1 == v9[1] )
          return v9[2];
        if ( *v9 == -4096 && v9[1] == -4096 )
          break;
        v10 = v7 + i;
        ++v7;
      }
      return 0;
    }
  }
  return v1;
}
