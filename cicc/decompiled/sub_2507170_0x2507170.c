// Function: sub_2507170
// Address: 0x2507170
//
__int64 __fastcall sub_2507170(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  int v8; // r11d
  unsigned int i; // eax
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // rax

  v2 = *a1;
  v3 = **a1;
  if ( !v3 )
    return 0;
  if ( !*((_BYTE *)v2 + 16) )
    return sub_BC1CD0(v3, &unk_4F81450, a2) + 8;
  v6 = *(unsigned int *)(v3 + 88);
  v7 = *(_QWORD *)(v3 + 72);
  if ( !(_DWORD)v6 )
    return 0;
  v8 = 1;
  for ( i = (v6 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v6 - 1) & v11 )
  {
    v10 = v7 + 24LL * i;
    if ( *(_UNKNOWN **)v10 == &unk_4F81450 && a2 == *(_QWORD *)(v10 + 8) )
      break;
    if ( *(_QWORD *)v10 == -4096 && *(_QWORD *)(v10 + 8) == -4096 )
      return 0;
    v11 = v8 + i;
    ++v8;
  }
  if ( v10 != v7 + 24 * v6 && (v12 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL)) != 0 )
    return v12 + 8;
  else
    return 0;
}
