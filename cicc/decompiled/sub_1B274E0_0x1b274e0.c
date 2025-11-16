// Function: sub_1B274E0
// Address: 0x1b274e0
//
bool __fastcall sub_1B274E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 *v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rdi
  int v8; // r9d
  unsigned int v9; // esi
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  int v16; // r10d

  v2 = *a1;
  if ( (*(_BYTE *)(*a1 + 8) & 1) != 0 )
  {
    v3 = v2 + 16;
    v4 = 256;
  }
  else
  {
    v3 = *(_QWORD *)(v2 + 16);
    v4 = 16LL * *(unsigned int *)(v2 + 24);
  }
  v5 = (__int64 *)(v3 + v4);
  v6 = sub_15E4F10(a2);
  if ( (*(_BYTE *)(v2 + 8) & 1) != 0 )
  {
    v7 = v2 + 16;
    v8 = 15;
  }
  else
  {
    v13 = *(unsigned int *)(v2 + 24);
    v7 = *(_QWORD *)(v2 + 16);
    if ( !(_DWORD)v13 )
      goto LABEL_11;
    v8 = v13 - 1;
  }
  v9 = v8 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v6 == *v10 )
    return v5 == v10;
  v15 = 1;
  while ( v11 != -8 )
  {
    v16 = v15 + 1;
    v9 = v8 & (v15 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v6 == *v10 )
      return v5 == v10;
    v15 = v16;
  }
  if ( (*(_BYTE *)(v2 + 8) & 1) != 0 )
  {
    v14 = 256;
    return v5 == (__int64 *)(v7 + v14);
  }
  v13 = *(unsigned int *)(v2 + 24);
LABEL_11:
  v14 = 16 * v13;
  return v5 == (__int64 *)(v7 + v14);
}
