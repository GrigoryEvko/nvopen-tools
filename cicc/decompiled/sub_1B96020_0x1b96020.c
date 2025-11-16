// Function: sub_1B96020
// Address: 0x1b96020
//
__int64 __fastcall sub_1B96020(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v11; // eax
  int v12; // edx
  int v13; // r9d

  if ( a3 <= 1 )
    return 0;
  v4 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v4 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
    {
LABEL_5:
      if ( v9 != (__int64 *)(v7 + 16 * v4) && !(unsigned __int8)sub_1B918B0(a1, a2, a3) )
      {
        LOBYTE(v11) = sub_1B95F70(a1, a2, a3);
        return v11 ^ 1u;
      }
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v13 = v12 + 1;
        v8 = (v4 - 1) & (v12 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_5;
        v12 = v13;
      }
    }
  }
  return 0;
}
