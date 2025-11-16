// Function: sub_13A2D40
// Address: 0x13a2d40
//
__int64 __fastcall sub_13A2D40(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 *v6; // rdi
  int v7; // edx
  unsigned int v8; // eax
  __int64 *v9; // rsi
  __int64 v10; // r8
  int v11; // esi
  int v12; // r9d

  sub_13A29A0(a1);
  if ( sub_13A0E30(a1 + 32, a2) )
    return 0;
  v4 = *(_QWORD *)(a1 + 336);
  v5 = *(unsigned int *)(a1 + 352);
  v6 = (__int64 *)(v4 + 24 * v5);
  if ( (_DWORD)v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v4 + 24LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_5:
      if ( v6 != v9 )
        return 0;
    }
    else
    {
      v11 = 1;
      while ( v10 != -8 )
      {
        v12 = v11 + 1;
        v8 = v7 & (v11 + v8);
        v9 = (__int64 *)(v4 + 24LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_5;
        v11 = v12;
      }
    }
  }
  return (unsigned int)sub_139F030(a2) ^ 1;
}
