// Function: sub_2B29CF0
// Address: 0x2b29cf0
//
__int64 __fastcall sub_2B29CF0(__int64 *a1, _BYTE *a2)
{
  __int64 v3; // r9
  char v4; // r8
  __int64 v5; // r10
  int v6; // r11d
  unsigned int v7; // ecx
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // edx
  int v17; // ebx
  __int64 v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdx

  if ( *a2 <= 0x1Cu )
    return 0;
  v3 = *a1;
  v4 = *(_BYTE *)(*a1 + 88) & 1;
  if ( v4 )
  {
    v5 = v3 + 96;
    v6 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(v3 + 104);
    v5 = *(_QWORD *)(v3 + 96);
    if ( !(_DWORD)v14 )
      goto LABEL_18;
    v6 = v14 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = v5 + 72LL * v7;
  v9 = *(_BYTE **)v8;
  if ( a2 != *(_BYTE **)v8 )
  {
    v16 = 1;
    if ( v9 != (_BYTE *)-4096LL )
    {
      while ( 1 )
      {
        v17 = v16 + 1;
        v7 = v6 & (v16 + v7);
        v18 = v5 + 72LL * v7;
        v19 = *(_BYTE **)v18;
        if ( a2 == *(_BYTE **)v18 )
          break;
        v16 = v17;
        if ( v19 == (_BYTE *)-4096LL )
        {
          if ( v4 )
            v24 = 288;
          else
            v24 = 72LL * *(unsigned int *)(v3 + 104);
          v18 = v5 + v24;
          break;
        }
      }
      v20 = 288;
      if ( !v4 )
        v20 = 72LL * *(unsigned int *)(v3 + 104);
      if ( v18 == v5 + v20 )
        return 0;
      v21 = *(__int64 **)(v18 + 8);
      v22 = &v21[*(unsigned int *)(v18 + 16)];
      if ( v22 == v21 )
        return 0;
      while ( 1 )
      {
        v23 = *v21;
        if ( *(_QWORD *)a1[1] == *(_QWORD *)(*v21 + 184) && *(_DWORD *)a1[2] == *(_DWORD *)(v23 + 192) )
          break;
        if ( v22 == ++v21 )
          return 0;
      }
      *(_QWORD *)a1[3] = v23;
      return 1;
    }
    if ( v4 )
    {
      v15 = 288;
      goto LABEL_19;
    }
    v14 = *(unsigned int *)(v3 + 104);
LABEL_18:
    v15 = 72 * v14;
LABEL_19:
    v8 = v5 + v15;
  }
  v10 = 288;
  if ( !v4 )
    v10 = 72LL * *(unsigned int *)(v3 + 104);
  if ( v8 == v5 + v10 )
    return 0;
  v11 = *(__int64 **)(v8 + 8);
  v12 = &v11[*(unsigned int *)(v8 + 16)];
  if ( v12 == v11 )
    return 0;
  while ( 1 )
  {
    v13 = *v11;
    if ( *(_QWORD *)a1[1] == *(_QWORD *)(*v11 + 184) && *(_DWORD *)a1[2] == *(_DWORD *)(v13 + 192) )
      break;
    if ( v12 == ++v11 )
      return 0;
  }
  *(_QWORD *)a1[3] = v13;
  return 1;
}
