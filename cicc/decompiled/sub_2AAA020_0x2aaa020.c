// Function: sub_2AAA020
// Address: 0x2aaa020
//
bool __fastcall sub_2AAA020(__int64 *a1, int *a2)
{
  int v3; // esi
  char v4; // r9
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // ebx
  unsigned int i; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax

  v3 = *a2;
  v4 = *((_BYTE *)a2 + 4);
  v5 = *a1;
  v6 = *(_QWORD *)a1[1];
  v7 = *(_QWORD *)(v5 + 40);
  v8 = *(unsigned int *)(v7 + 408);
  v9 = *(_QWORD *)(v7 + 392);
  if ( !(_DWORD)v8 )
    return *(_DWORD *)(v9 + (v8 << 6) + 16) == 7;
  v10 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)(v4 == 0) + 37 * v3 - 1)
              | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v4 == 0) + 37 * v3 - 1))); ; i = (v8 - 1) & v13 )
  {
    v12 = v9 + ((unsigned __int64)i << 6);
    if ( v6 == *(_QWORD *)v12 && v3 == *(_DWORD *)(v12 + 8) && v4 == *(_BYTE *)(v12 + 12) )
      return *(_DWORD *)(v12 + 16) == 7;
    if ( *(_QWORD *)v12 == -4096 && *(_DWORD *)(v12 + 8) == -1 && *(_BYTE *)(v12 + 12) )
      break;
    v13 = v10 + i;
    ++v10;
  }
  return *(_DWORD *)(v9 + (v8 << 6) + 16) == 7;
}
