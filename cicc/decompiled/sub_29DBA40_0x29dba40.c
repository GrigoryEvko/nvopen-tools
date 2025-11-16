// Function: sub_29DBA40
// Address: 0x29dba40
//
__int64 __fastcall sub_29DBA40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // eax
  int v5; // edx
  unsigned int v6; // eax
  __int64 v7; // rdi
  int v9; // r8d

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    v4 = *(_DWORD *)(v2 + 24);
    if ( v4 )
    {
      v5 = v4 - 1;
      v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = *(_QWORD *)(v3 + 8LL * v6);
      if ( a2 == v7 )
        return 1;
      v9 = 1;
      while ( v7 != -4096 )
      {
        v6 = v5 & (v9 + v6);
        v7 = *(_QWORD *)(v3 + 8LL * v6);
        if ( a2 == v7 )
          return 1;
        ++v9;
      }
    }
  }
  return 0;
}
