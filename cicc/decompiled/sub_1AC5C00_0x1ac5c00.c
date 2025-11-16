// Function: sub_1AC5C00
// Address: 0x1ac5c00
//
__int64 __fastcall sub_1AC5C00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r8
  __int64 v5; // r8
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v11; // eax
  int v12; // r9d

  v2 = a1[6];
  if ( v2 == a1[7] )
    v2 = *(_QWORD *)(a1[9] - 8LL) + 512LL;
  v3 = *(_DWORD *)(v2 - 8);
  v4 = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(v2 - 24);
    v6 = v3 - 1;
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
      return v8[1];
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v12 = v11 + 1;
        v7 = v6 & (v11 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          return v8[1];
        v11 = v12;
      }
      return 0;
    }
  }
  return v4;
}
