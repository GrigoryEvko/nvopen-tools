// Function: sub_161E8E0
// Address: 0x161e8e0
//
__int64 __fastcall sub_161E8E0(__int64 a1)
{
  __int64 v2; // rcx
  __int64 result; // rax
  int v4; // edx
  __int64 v5; // rdi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rsi
  int v10; // eax
  int v11; // r8d

  v2 = *(_QWORD *)sub_16498A0(a1);
  result = 0;
  v4 = *(_DWORD *)(v2 + 424);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v2 + 408);
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
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
        if ( a1 == *v8 )
          return v8[1];
        v10 = v11;
      }
      return 0;
    }
  }
  return result;
}
