// Function: sub_1629050
// Address: 0x1629050
//
__int64 __fastcall sub_1629050(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  int v4; // edx
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // rdi
  int v11; // edx
  int v12; // r9d

  v2 = sub_1628D40(a1, a2);
  v3 = 0;
  v4 = *(_DWORD *)(*a1 + 456);
  if ( v4 )
  {
    v5 = *(_QWORD *)(*a1 + 440);
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( v2 == *v8 )
    {
      return v8[1];
    }
    else
    {
      v11 = 1;
      while ( v9 != -4 )
      {
        v12 = v11 + 1;
        v7 = v6 & (v11 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( v2 == *v8 )
          return v8[1];
        v11 = v12;
      }
      return 0;
    }
  }
  return v3;
}
