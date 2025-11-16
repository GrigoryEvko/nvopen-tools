// Function: sub_21076B0
// Address: 0x21076b0
//
__int64 __fastcall sub_21076B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  int v3; // eax
  __int64 v4; // rdi
  int v5; // edx
  unsigned int v6; // eax
  __int64 v7; // rcx
  int i; // r8d

  v2 = 0;
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 24LL);
  if ( v3 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v5 = v3 - 1;
    v2 = 1;
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = *(_QWORD *)(v4 + 16LL * v6);
    if ( v7 != a2 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v7 == -8 )
          return 0;
        v6 = v5 & (i + v6);
        v7 = *(_QWORD *)(v4 + 16LL * v6);
        if ( a2 == v7 )
          break;
      }
      return 1;
    }
  }
  return v2;
}
