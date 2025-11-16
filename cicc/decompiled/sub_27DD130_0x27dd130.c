// Function: sub_27DD130
// Address: 0x27dd130
//
__int64 __fastcall sub_27DD130(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r9
  __int64 v4; // rdx
  __int64 v5; // r8
  int v6; // ebx
  unsigned int i; // eax
  __int64 v8; // rcx
  unsigned int v9; // eax

  if ( *((_BYTE *)a1 + 80) )
    return a1[9];
  v2 = a1[1];
  v3 = *a1;
  v4 = *(unsigned int *)(v2 + 88);
  v5 = *(_QWORD *)(v2 + 72);
  if ( (_DWORD)v4 )
  {
    v6 = 1;
    for ( i = (v4 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F8E5A8 >> 9) ^ ((unsigned int)&unk_4F8E5A8 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)))); ; i = (v4 - 1) & v9 )
    {
      v8 = v5 + 24LL * i;
      if ( *(_UNKNOWN **)v8 == &unk_4F8E5A8 && v3 == *(_QWORD *)(v8 + 8) )
        break;
      if ( *(_QWORD *)v8 == -4096 && *(_QWORD *)(v8 + 8) == -4096 )
        goto LABEL_14;
      v9 = v6 + i;
      ++v6;
    }
    if ( v8 == v5 + 24 * v4 )
    {
LABEL_14:
      result = 0;
      goto LABEL_12;
    }
    result = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL);
    if ( result )
      result += 8;
LABEL_12:
    a1[9] = result;
    *((_BYTE *)a1 + 80) = 1;
  }
  else
  {
    *((_BYTE *)a1 + 80) = 1;
    a1[9] = 0;
    return 0;
  }
  return result;
}
