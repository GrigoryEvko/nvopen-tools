// Function: sub_1E80290
// Address: 0x1e80290
//
__int64 __fastcall sub_1E80290(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  int v4; // edx
  __int64 v5; // r8
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  int v11; // r9d

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 440) + 264LL);
  result = 0;
  v4 = *(_DWORD *)(v2 + 256);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v2 + 240);
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
      return v8[1];
    }
    else
    {
      v10 = 1;
      while ( v9 != -8 )
      {
        v11 = v10 + 1;
        v7 = v6 & (v10 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          return v8[1];
        v10 = v11;
      }
      return 0;
    }
  }
  return result;
}
