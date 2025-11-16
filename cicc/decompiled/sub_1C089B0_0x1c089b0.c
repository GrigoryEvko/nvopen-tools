// Function: sub_1C089B0
// Address: 0x1c089b0
//
__int64 __fastcall sub_1C089B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 *v5; // r9
  int v6; // edx
  unsigned int v7; // eax
  __int64 *v8; // rsi
  int v10; // esi
  int v11; // r10d

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(unsigned int *)(a2 + 24);
  LODWORD(v4) = 0;
  v5 = (__int64 *)(v2 + 8 * v3);
  if ( (_DWORD)v3 )
  {
    v6 = v3 - 1;
    v7 = v6 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v2 + 8LL * v7);
    v4 = *v8;
    if ( a1 == *v8 )
    {
LABEL_3:
      LOBYTE(v4) = v5 != v8;
    }
    else
    {
      v10 = 1;
      while ( v4 != -8 )
      {
        v11 = v10 + 1;
        v7 = v6 & (v10 + v7);
        v8 = (__int64 *)(v2 + 8LL * v7);
        v4 = *v8;
        if ( a1 == *v8 )
          goto LABEL_3;
        v10 = v11;
      }
      LODWORD(v4) = 0;
    }
  }
  return (unsigned int)v4;
}
