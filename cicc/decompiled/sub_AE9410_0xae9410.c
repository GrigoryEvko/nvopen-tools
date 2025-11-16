// Function: sub_AE9410
// Address: 0xae9410
//
__int64 __fastcall sub_AE9410(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // r8
  int v9; // eax
  int v10; // r10d

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v1 = (_QWORD *)*v1;
  v2 = (_QWORD *)*v1;
  v3 = *((unsigned int *)v2 + 820);
  v4 = v2[408];
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v6 = (__int64 *)(v4 + 32LL * v5);
    v7 = *v6;
    if ( a1 == *v6 )
    {
LABEL_5:
      if ( v6 != (__int64 *)(v4 + 32 * v3) )
        return v6[1];
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v10 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 32LL * v5);
        v7 = *v6;
        if ( a1 == *v6 )
          goto LABEL_5;
        v9 = v10;
      }
    }
  }
  return 0;
}
