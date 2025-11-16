// Function: sub_2B147B0
// Address: 0x2b147b0
//
bool __fastcall sub_2B147B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // rdx
  int v8; // esi
  unsigned int v9; // eax
  __int64 *v10; // rcx
  __int64 v11; // r8
  int v13; // ecx
  __int64 v14; // rdi
  _BYTE *v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  _BYTE *v20; // rdi
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rax
  int v26; // r9d

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 != 86 )
  {
LABEL_2:
    v4 = *(_QWORD *)(a1[1] + 3272);
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      v5 = v4 + 16;
      v7 = (__int64 *)(v4 + 48);
      v8 = 3;
    }
    else
    {
      v5 = *(_QWORD *)(v4 + 16);
      v6 = *(unsigned int *)(v4 + 24);
      v7 = (__int64 *)(v5 + 8 * v6);
      if ( !(_DWORD)v6 )
        return 0;
      v8 = v6 - 1;
    }
    v9 = v8 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v10 = (__int64 *)(v5 + 8LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
      return v7 != v10;
    v13 = 1;
    while ( v11 != -4096 )
    {
      v26 = v13 + 1;
      v9 = v8 & (v13 + v9);
      v10 = (__int64 *)(v5 + 8LL * v9);
      v11 = *v10;
      if ( *v10 == v3 )
        return v7 != v10;
      v13 = v26;
    }
    return 0;
  }
  v14 = *(_QWORD *)(v3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  if ( !sub_BCAC40(v14, 1)
    || *(_BYTE *)v3 != 57
    && (*(_BYTE *)v3 != 86
     || *(_QWORD *)(*(_QWORD *)(v3 - 96) + 8LL) != *(_QWORD *)(v3 + 8)
     || (v15 = *(_BYTE **)(v3 - 32), *v15 > 0x15u)
     || !sub_AC30F0((__int64)v15)) )
  {
    v3 = *(_QWORD *)(a2 + 24);
    if ( *(_BYTE *)v3 <= 0x1Cu )
      goto LABEL_2;
    v16 = *(_QWORD *)(v3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    if ( !sub_BCAC40(v16, 1) )
      goto LABEL_29;
    if ( *(_BYTE *)v3 != 58 )
    {
      if ( *(_BYTE *)v3 != 86 )
        goto LABEL_29;
      v19 = *(_QWORD *)(v3 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(v3 - 96) + 8LL) != v19 )
        goto LABEL_29;
      v20 = *(_BYTE **)(v3 - 64);
      if ( *v20 > 0x15u || !sub_AD7A80(v20, 1, v19, v17, v18) )
        goto LABEL_29;
    }
  }
  if ( (unsigned int)sub_BD2910(a2) )
  {
LABEL_29:
    v3 = *(_QWORD *)(a2 + 24);
    goto LABEL_2;
  }
  v23 = *a1;
  v24 = *(_QWORD *)(a2 + 24);
  v25 = *(unsigned int *)(*a1 + 8);
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(*a1, (const void *)(v23 + 16), v25 + 1, 8u, v21, v22);
    v25 = *(unsigned int *)(v23 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v23 + 8 * v25) = v24;
  ++*(_DWORD *)(v23 + 8);
  return 0;
}
