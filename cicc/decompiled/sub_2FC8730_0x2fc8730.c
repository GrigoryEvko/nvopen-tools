// Function: sub_2FC8730
// Address: 0x2fc8730
//
__int64 __fastcall sub_2FC8730(_DWORD *a1, int a2)
{
  int v2; // r12d
  __int64 v3; // rbx
  __int64 v4; // rdi
  unsigned int v5; // r12d
  __int64 v6; // r13
  __int64 v7; // rbx

  v2 = a1[2];
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v4 = *(_QWORD *)a1;
  v5 = *(_DWORD *)(v3 + 40LL * (unsigned int)(v2 + 2) + 24) + v2 + 4;
  v6 = v3 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF);
  v7 = v3 + 40LL * (unsigned int)sub_2E88FE0(v4);
  if ( v6 != v7 )
  {
    while ( (unsigned int)sub_2EAB0A0(v7) < v5 )
    {
      if ( !*(_BYTE *)v7 && a2 == *(_DWORD *)(v7 + 8) )
        return 0;
      v7 += 40;
      if ( v6 == v7 )
        return 1;
    }
  }
  return 1;
}
