// Function: sub_2485740
// Address: 0x2485740
//
__int64 __fastcall sub_2485740(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rdi
  int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // rcx
  int v8; // r8d

  v2 = *(_DWORD *)(*(_QWORD *)a1 + 24LL);
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = v4 & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
    v6 = *(_QWORD *)(v3 + 24LL * v5);
    if ( v6 == a2 )
      return 1;
    v8 = 1;
    while ( v6 != -1 )
    {
      v5 = v4 & (v8 + v5);
      v6 = *(_QWORD *)(v3 + 24LL * v5);
      if ( a2 == v6 )
        return 1;
      ++v8;
    }
  }
  return 0;
}
