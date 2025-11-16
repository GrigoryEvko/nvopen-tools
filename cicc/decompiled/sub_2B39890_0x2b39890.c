// Function: sub_2B39890
// Address: 0x2b39890
//
bool __fastcall sub_2B39890(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v4; // rdx
  __int64 *v5; // rcx
  __int64 v6; // rdx
  char v8; // r11
  __int64 v9; // r8
  __int64 *v10; // r9
  int v11; // r13d
  unsigned int v12; // esi
  __int64 *v13; // rdi
  __int64 v14; // r14
  __int64 v15; // rdi
  int v16; // edi
  int v17; // r15d

  v2 = *(__int64 **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 28) )
    v4 = *(unsigned int *)(a1 + 20);
  else
    v4 = *(unsigned int *)(a1 + 16);
  v5 = &v2[v4];
  if ( v5 == v2 )
    return v2 != v5;
  while ( 1 )
  {
    v6 = *v2;
    if ( (unsigned __int64)*v2 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( ++v2 == v5 )
      return v2 != v5;
  }
  if ( v2 == v5 )
    return v2 != v5;
  v8 = *(_BYTE *)(a2 + 8) & 1;
  if ( !v8 )
    goto LABEL_18;
LABEL_9:
  v9 = a2 + 16;
  v10 = (__int64 *)(a2 + 48);
  v11 = 3;
LABEL_10:
  v12 = v11 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v13 = (__int64 *)(v9 + 8LL * v12);
  v14 = *v13;
  if ( *v13 == v6 )
  {
LABEL_11:
    if ( v13 == v10 )
      goto LABEL_14;
  }
  else
  {
    v16 = 1;
    while ( v14 != -4096 )
    {
      v17 = v16 + 1;
      v12 = v11 & (v16 + v12);
      v13 = (__int64 *)(v9 + 8LL * v12);
      v14 = *v13;
      if ( *v13 == v6 )
        goto LABEL_11;
      v16 = v17;
    }
LABEL_14:
    while ( ++v2 != v5 )
    {
      v6 = *v2;
      if ( (unsigned __int64)*v2 < 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v2 == v5 )
          return v2 != v5;
        if ( v8 )
          goto LABEL_9;
LABEL_18:
        v9 = *(_QWORD *)(a2 + 16);
        v15 = *(unsigned int *)(a2 + 24);
        v10 = (__int64 *)(v9 + 8 * v15);
        if ( (_DWORD)v15 )
        {
          v11 = v15 - 1;
          goto LABEL_10;
        }
      }
    }
  }
  return v2 != v5;
}
