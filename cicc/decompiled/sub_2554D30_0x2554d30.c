// Function: sub_2554D30
// Address: 0x2554d30
//
__int64 __fastcall sub_2554D30(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r8
  __int64 v4; // rdx
  __int64 v5; // rdi
  int v6; // r11d
  unsigned int i; // eax
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v11; // rax

  v3 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return 0;
  if ( !*(_BYTE *)(a1 + 16) && !a3 )
    return sub_BC1CD0(v3, &unk_4F81450, a2) + 8;
  v4 = *(unsigned int *)(v3 + 88);
  v5 = *(_QWORD *)(v3 + 72);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = 1;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v4 - 1) & v9 )
  {
    v8 = v5 + 24LL * i;
    if ( *(_UNKNOWN **)v8 == &unk_4F81450 && a2 == *(_QWORD *)(v8 + 8) )
      break;
    if ( *(_QWORD *)v8 == -4096 && *(_QWORD *)(v8 + 8) == -4096 )
      return 0;
    v9 = v6 + i;
    ++v6;
  }
  if ( v8 == v5 + 24 * v4 )
    return 0;
  v11 = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL);
  if ( !v11 )
    return 0;
  return v11 + 8;
}
