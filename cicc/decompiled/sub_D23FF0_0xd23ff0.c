// Function: sub_D23FF0
// Address: 0xd23ff0
//
_QWORD *__fastcall sub_D23FF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rbx
  __int64 v9; // r13
  _QWORD *i; // r12
  _QWORD *result; // rax
  _QWORD *v12; // r14
  int v13; // eax
  int v14; // r10d

  v2 = *(unsigned int *)(a1 + 120);
  v3 = *(_QWORD *)(a1 + 104);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
      goto LABEL_3;
    v13 = 1;
    while ( v6 != -4096 )
    {
      v14 = v13 + 1;
      v4 = (v2 - 1) & (v13 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        goto LABEL_3;
      v13 = v14;
    }
  }
  v5 = (__int64 *)(v3 + 16 * v2);
LABEL_3:
  v7 = v5[1];
  v8 = *(_QWORD **)(v7 + 24);
  v9 = v7 + 24;
  for ( i = &v8[*(unsigned int *)(v7 + 32)]; i != v8; ++v8 )
  {
    if ( (*v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL) )
      break;
  }
  result = (_QWORD *)sub_D23C30(v9);
  v12 = result;
  while ( v8 != v12 )
  {
    if ( (*v8 & 4) != 0 )
      result = (_QWORD *)sub_D23D60(v9, *v8 & 0xFFFFFFFFFFFFFFF8LL, 0);
    while ( i != ++v8 )
    {
      result = (_QWORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL);
      if ( result )
      {
        if ( *result )
          break;
      }
    }
  }
  return result;
}
