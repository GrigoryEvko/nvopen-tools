// Function: sub_2B22B70
// Address: 0x2b22b70
//
__int64 __fastcall sub_2B22B70(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r9
  int v4; // ecx
  unsigned int v5; // r10d
  __int64 v6; // r8
  __int64 v7; // rcx
  int v8; // edx
  int v9; // edx
  __int64 v10; // r8
  __int64 result; // rax
  unsigned int v12; // edi
  __int64 v13; // rcx
  int v14; // ecx
  int v15; // r11d
  __int64 v16; // rcx
  int v17; // eax
  int v18; // r9d
  __int64 v19; // rax

  v2 = *a1;
  if ( (*(_BYTE *)(*a1 + 88) & 1) != 0 )
  {
    v3 = v2 + 96;
    v4 = 3;
  }
  else
  {
    v14 = *(_DWORD *)(v2 + 104);
    v3 = *(_QWORD *)(v2 + 96);
    if ( !v14 )
      goto LABEL_11;
    v4 = v14 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v6 = *(_QWORD *)(v3 + 72LL * v5);
  if ( a2 == v6 )
  {
LABEL_4:
    v7 = a1[1];
    v8 = *(_DWORD *)(v7 + 24);
    if ( v8 )
    {
      v9 = v8 - 1;
      v10 = *(_QWORD *)(v7 + 8);
      result = 1;
      v12 = v9 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
      v13 = *(_QWORD *)(v10 + 16LL * v12);
      if ( a2 == v13 )
        return result;
      v17 = 1;
      while ( v13 != -4096 )
      {
        v18 = v17 + 1;
        v19 = v9 & (v12 + v17);
        v12 = v19;
        v13 = *(_QWORD *)(v10 + 16 * v19);
        if ( a2 == v13 )
          return 1;
        v17 = v18;
      }
    }
    return 0;
  }
  v15 = 1;
  while ( v6 != -4096 )
  {
    v5 = v4 & (v15 + v5);
    v6 = *(_QWORD *)(v3 + 72LL * v5);
    if ( a2 == v6 )
      goto LABEL_4;
    ++v15;
  }
LABEL_11:
  result = 1;
  if ( *(_BYTE *)a2 == 90 )
  {
    v16 = *(_QWORD *)(a2 + 16);
    if ( v16 )
    {
      if ( !*(_QWORD *)(v16 + 8) )
        return (unsigned int)sub_B19060(v2 + 768, a2, v2, v16) ^ 1;
    }
  }
  return result;
}
