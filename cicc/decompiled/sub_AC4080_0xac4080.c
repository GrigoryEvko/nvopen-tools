// Function: sub_AC4080
// Address: 0xac4080
//
__int64 __fastcall sub_AC4080(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rcx
  __int64 *v3; // rax
  __int64 v4; // rdx
  int v5; // r10d
  __int64 v6; // r9
  int v7; // r10d
  int v8; // ebx
  unsigned int i; // eax
  _QWORD *v10; // rsi
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 32);
  v3 = **(__int64 ***)(v1 + 8);
  v4 = *v3;
  v5 = *(_DWORD *)(*v3 + 2016);
  v6 = *(_QWORD *)(*v3 + 2000);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = 1;
    for ( i = v7
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)
                | ((unsigned __int64)(((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4)))); ; i = v7 & v11 )
    {
      v10 = (_QWORD *)(v6 + 24LL * i);
      if ( v1 == *v10 && v2 == v10[1] )
        break;
      if ( *v10 == -4096 && v10[1] == -4096 )
        goto LABEL_8;
      v11 = v8 + i;
      ++v8;
    }
    *v10 = -8192;
    v10[1] = -8192;
    --*(_DWORD *)(v4 + 2008);
    ++*(_DWORD *)(v4 + 2012);
    v2 = *(_QWORD *)(a1 - 32);
  }
LABEL_8:
  v12 = *(unsigned __int16 *)(v2 + 2);
  v13 = v12 + 0x7FFF;
  LOWORD(v12) = v12 & 0x8000;
  LOWORD(v13) = v13 & 0x7FFF;
  result = v13 | v12;
  *(_WORD *)(v2 + 2) = result;
  return result;
}
