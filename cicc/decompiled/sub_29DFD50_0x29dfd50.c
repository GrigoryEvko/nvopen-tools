// Function: sub_29DFD50
// Address: 0x29dfd50
//
char *__fastcall sub_29DFD50(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r12
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 *v6; // r12
  __int64 *i; // r12
  __int64 v8; // rax
  char *result; // rax
  __int64 v10; // r12

  v1 = *(__int64 **)a1;
  v2 = 8LL * *(unsigned int *)(a1 + 8);
  v3 = (__int64 *)(*(_QWORD *)a1 + v2);
  v4 = v2 >> 3;
  v5 = v2 >> 5;
  if ( v5 )
  {
    v6 = &v1[4 * v5];
    while ( !sub_AA4F10(*(_QWORD *)(*v1 + 40)) )
    {
      if ( sub_AA4F10(*(_QWORD *)(v1[1] + 40)) )
      {
        ++v1;
        goto LABEL_8;
      }
      if ( sub_AA4F10(*(_QWORD *)(v1[2] + 40)) )
      {
        v1 += 2;
        goto LABEL_8;
      }
      if ( sub_AA4F10(*(_QWORD *)(v1[3] + 40)) )
      {
        v1 += 3;
        goto LABEL_8;
      }
      v1 += 4;
      if ( v6 == v1 )
      {
        v4 = v3 - v1;
        goto LABEL_18;
      }
    }
    goto LABEL_8;
  }
LABEL_18:
  if ( v4 == 2 )
  {
LABEL_24:
    if ( sub_AA4F10(*(_QWORD *)(*v1 + 40)) )
      goto LABEL_8;
    ++v1;
    goto LABEL_26;
  }
  if ( v4 == 3 )
  {
    if ( sub_AA4F10(*(_QWORD *)(*v1 + 40)) )
      goto LABEL_8;
    ++v1;
    goto LABEL_24;
  }
  if ( v4 != 1 )
  {
LABEL_21:
    v1 = v3;
    goto LABEL_14;
  }
LABEL_26:
  if ( !sub_AA4F10(*(_QWORD *)(*v1 + 40)) )
    goto LABEL_21;
LABEL_8:
  if ( v3 != v1 )
  {
    for ( i = v1 + 1; v3 != i; *(v1 - 1) = v8 )
    {
      while ( sub_AA4F10(*(_QWORD *)(*i + 40)) )
      {
        if ( v3 == ++i )
          goto LABEL_14;
      }
      v8 = *i++;
      ++v1;
    }
  }
LABEL_14:
  result = *(char **)a1;
  v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v3;
  if ( v3 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v1, v3, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v3);
    result = *(char **)a1;
  }
  *(_DWORD *)(a1 + 8) = ((char *)v1 + v10 - result) >> 3;
  return result;
}
