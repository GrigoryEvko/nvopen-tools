// Function: sub_19E56C0
// Address: 0x19e56c0
//
__int64 __fastcall sub_19E56C0(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v3; // ecx
  __int64 v4; // r8
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v9; // eax
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 24);
  if ( v2 )
  {
    v3 = v2 - 1;
    v4 = *(_QWORD *)(a1 + 8);
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
      return v6[1];
    v9 = 1;
    while ( v7 != -8 )
    {
      v10 = v9 + 1;
      v5 = v3 & (v9 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( *v6 == a2 )
        return v6[1];
      v9 = v10;
    }
  }
  return 0;
}
