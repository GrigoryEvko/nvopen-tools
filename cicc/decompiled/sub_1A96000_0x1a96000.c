// Function: sub_1A96000
// Address: 0x1a96000
//
__int64 __fastcall sub_1A96000(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  int v3; // edx
  __int64 *v5; // r8
  unsigned int v6; // r9d
  __int64 *v7; // rcx

  v2 = *(__int64 **)(a1 + 8);
  if ( *(__int64 **)(a1 + 16) != v2 )
    goto LABEL_2;
  v5 = &v2[*(unsigned int *)(a1 + 28)];
  v6 = *(_DWORD *)(a1 + 28);
  if ( v2 == v5 )
    goto LABEL_12;
  v7 = 0;
  do
  {
    if ( a2 == *v2 )
      return 1;
    if ( *v2 == -2 )
      v7 = v2;
    ++v2;
  }
  while ( v5 != v2 );
  if ( !v7 )
  {
LABEL_12:
    if ( v6 >= *(_DWORD *)(a1 + 24) )
    {
LABEL_2:
      sub_16CCBA0(a1, a2);
      return v3 ^ 1u;
    }
    *(_DWORD *)(a1 + 28) = v6 + 1;
    *v5 = a2;
    ++*(_QWORD *)a1;
    return 0;
  }
  else
  {
    *v7 = a2;
    --*(_DWORD *)(a1 + 32);
    ++*(_QWORD *)a1;
    return 0;
  }
}
