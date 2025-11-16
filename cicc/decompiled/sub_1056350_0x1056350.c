// Function: sub_1056350
// Address: 0x1056350
//
__int64 __fastcall sub_1056350(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  int v3; // eax
  int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // rdi
  int v8; // r8d

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 248LL);
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 264LL);
  if ( v3 )
  {
    v4 = v3 - 1;
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = *(_QWORD *)(v2 + 8LL * v5);
    if ( a2 == v6 )
      return 1;
    v8 = 1;
    while ( v6 != -4096 )
    {
      v5 = v4 & (v8 + v5);
      v6 = *(_QWORD *)(v2 + 8LL * v5);
      if ( a2 == v6 )
        return 1;
      ++v8;
    }
  }
  return 0;
}
