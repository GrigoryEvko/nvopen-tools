// Function: sub_397FB50
// Address: 0x397fb50
//
__int64 __fastcall sub_397FB50(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 result; // rax
  int v4; // ecx
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 408);
  result = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(a1 + 392);
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
      return v7[1];
    }
    else
    {
      v9 = 1;
      while ( v8 != -8 )
      {
        v10 = v9 + 1;
        v6 = v4 & (v9 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          return v7[1];
        v9 = v10;
      }
      return 0;
    }
  }
  return result;
}
