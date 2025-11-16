// Function: sub_16CC9F0
// Address: 0x16cc9f0
//
_QWORD *__fastcall sub_16CC9F0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdi
  int v4; // edx
  unsigned int v5; // eax
  _QWORD *v6; // r8
  __int64 v7; // rcx
  _QWORD *v9; // r10
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 16);
  v4 = v2 - 1;
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (_QWORD *)(v3 + 8LL * v5);
  v7 = *v6;
  if ( *v6 == -1 )
    return (_QWORD *)(v3 + 8LL * (v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  v9 = 0;
  v10 = 1;
  if ( a2 == v7 )
    return (_QWORD *)(v3 + 8LL * (v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  while ( 1 )
  {
    if ( v7 != -2 || v9 )
      v6 = v9;
    v5 = v4 & (v10 + v5);
    v7 = *(_QWORD *)(v3 + 8LL * v5);
    if ( v7 == -1 )
      break;
    v9 = v6;
    ++v10;
    v6 = (_QWORD *)(v3 + 8LL * v5);
    if ( a2 == v7 )
      return v6;
  }
  if ( !v6 )
    return (_QWORD *)(v3 + 8LL * v5);
  return v6;
}
