// Function: sub_161C9B0
// Address: 0x161c9b0
//
__int64 __fastcall sub_161C9B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // rcx
  unsigned int v5; // r11d
  __int64 *v6; // r9
  __int64 v7; // rdx
  int i; // r12d
  __int64 v9; // r8
  __int64 v10; // r13
  _QWORD *v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // r8
  _QWORD *v15; // rdi
  _QWORD *v16; // rdx
  _QWORD *v17; // r8

  v2 = *(unsigned int *)(a1 + 24);
  v3 = 0;
  if ( !(_DWORD)v2 )
    return 0;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (v2 - 1) & *(_DWORD *)(a2 + 32);
  v6 = (__int64 *)(v4 + 8LL * v5);
  v7 = *v6;
  if ( *v6 == -8 )
    return v3;
  for ( i = 1; ; ++i )
  {
    if ( v7 == -16 || *(_DWORD *)(a2 + 32) != *(_DWORD *)(v7 + 4) )
      goto LABEL_6;
    v9 = *(_QWORD *)(a2 + 8);
    v10 = *(unsigned int *)(v7 + 8);
    if ( v9 )
    {
      if ( v9 == v10 )
      {
        v11 = *(_QWORD **)a2;
        v12 = (_QWORD *)(v7 - 8 * v9);
        v13 = *(_QWORD *)a2 + 8 * v9;
        while ( *v11 == *v12 )
        {
          ++v11;
          ++v12;
          if ( (_QWORD *)v13 == v11 )
            goto LABEL_13;
        }
      }
      goto LABEL_6;
    }
    if ( *(_QWORD *)(a2 + 24) != v10 )
      goto LABEL_6;
    v15 = *(_QWORD **)(a2 + 16);
    v16 = (_QWORD *)(v7 - 8 * v10);
    v17 = &v15[v10];
    if ( v17 == v15 )
      break;
    while ( *v15 == *v16 )
    {
      ++v15;
      ++v16;
      if ( v17 == v15 )
        goto LABEL_13;
    }
LABEL_6:
    v5 = (v2 - 1) & (i + v5);
    v6 = (__int64 *)(v4 + 8LL * v5);
    v7 = *v6;
    if ( *v6 == -8 )
      return 0;
  }
LABEL_13:
  if ( v6 != (__int64 *)(v4 + 8 * v2) )
    return *v6;
  return 0;
}
