// Function: sub_25EFEE0
// Address: 0x25efee0
//
__int64 __fastcall sub_25EFEE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rdi
  int v4; // r11d
  unsigned int i; // eax
  __int64 v6; // rcx
  unsigned int v7; // eax
  __int64 result; // rax

  v2 = *(unsigned int *)(*(_QWORD *)a1 + 88LL);
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
  if ( !(_DWORD)v2 )
    return 0;
  v4 = 1;
  for ( i = (v2 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v2 - 1) & v7 )
  {
    v6 = v3 + 24LL * i;
    if ( *(_UNKNOWN **)v6 == &unk_4F86630 && a2 == *(_QWORD *)(v6 + 8) )
      break;
    if ( *(_QWORD *)v6 == -4096 && *(_QWORD *)(v6 + 8) == -4096 )
      return 0;
    v7 = v4 + i;
    ++v4;
  }
  if ( v6 == v3 + 24 * v2 )
    return 0;
  result = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL);
  if ( result )
    result += 8;
  return result;
}
