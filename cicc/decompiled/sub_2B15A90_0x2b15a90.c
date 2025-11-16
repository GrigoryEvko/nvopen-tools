// Function: sub_2B15A90
// Address: 0x2b15a90
//
bool __fastcall sub_2B15A90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v6; // r14
  __int64 v7; // rdi
  int v8; // edx
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rsi
  int v12; // edx
  int v13; // r8d
  _QWORD *v14; // rax
  _QWORD *v15; // rdx

  v4 = a1;
  if ( a2 == a1 )
    return a2 != v4;
  v6 = a3 + 768;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v4 + 24);
    if ( (*(_BYTE *)(a3 + 88) & 1) != 0 )
    {
      v7 = a3 + 96;
      v8 = 3;
    }
    else
    {
      v12 = *(_DWORD *)(a3 + 104);
      v7 = *(_QWORD *)(a3 + 96);
      if ( !v12 )
        goto LABEL_11;
      v8 = v12 - 1;
    }
    v9 = v8 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v10 = *(_QWORD *)(v7 + 72LL * v9);
    if ( v11 == v10 )
      goto LABEL_5;
    v13 = 1;
    while ( v10 != -4096 )
    {
      v9 = v8 & (v13 + v9);
      v10 = *(_QWORD *)(v7 + 72LL * v9);
      if ( v11 == v10 )
        goto LABEL_5;
      ++v13;
    }
LABEL_11:
    if ( !*(_BYTE *)(a3 + 796) )
      break;
    v14 = *(_QWORD **)(a3 + 776);
    v15 = &v14[*(unsigned int *)(a3 + 788)];
    if ( v14 == v15 )
      return a2 != v4;
    while ( v11 != *v14 )
    {
      if ( v15 == ++v14 )
        return a2 != v4;
    }
LABEL_5:
    v4 = *(_QWORD *)(v4 + 8);
    if ( a2 == v4 )
      return a2 != v4;
  }
  if ( sub_C8CA60(v6, v11) )
    goto LABEL_5;
  return a2 != v4;
}
