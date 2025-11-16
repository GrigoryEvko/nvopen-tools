// Function: sub_29DB840
// Address: 0x29db840
//
_QWORD *__fastcall sub_29DB840(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  const void *v6; // r13
  size_t v7; // r14
  _QWORD *v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  const void *v14; // r15
  size_t v15; // r13

  v3 = a1;
  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v4 <= 0 )
  {
LABEL_19:
    switch ( v5 )
    {
      case 2LL:
        v14 = *(const void **)a3;
        v15 = *(_QWORD *)(a3 + 8);
        break;
      case 3LL:
        v14 = *(const void **)a3;
        v15 = *(_QWORD *)(a3 + 8);
        if ( *(_QWORD *)(*v3 + 32LL) == v15
          && (!v15 || !memcmp(*(const void **)(*v3 + 24LL), *(const void **)a3, *(_QWORD *)(a3 + 8))) )
        {
          return v3;
        }
        ++v3;
        break;
      case 1LL:
        v14 = *(const void **)a3;
        v15 = *(_QWORD *)(a3 + 8);
LABEL_28:
        if ( v15 == *(_QWORD *)(*v3 + 32LL) )
        {
          if ( v15 && memcmp(*(const void **)(*v3 + 24LL), v14, v15) )
            return (_QWORD *)a2;
          return v3;
        }
        return (_QWORD *)a2;
      default:
        return (_QWORD *)a2;
    }
    if ( v15 == *(_QWORD *)(*v3 + 32LL) && (!v15 || !memcmp(*(const void **)(*v3 + 24LL), v14, v15)) )
      return v3;
    ++v3;
    goto LABEL_28;
  }
  v6 = *(const void **)a3;
  v7 = *(_QWORD *)(a3 + 8);
  v8 = &a1[4 * v4];
  while ( *(_QWORD *)(*v3 + 32LL) != v7 || v7 && memcmp(*(const void **)(*v3 + 24LL), v6, v7) )
  {
    v9 = v3[1];
    v10 = v3 + 1;
    if ( *(_QWORD *)(v9 + 32) == v7 && (!v7 || !memcmp(*(const void **)(v9 + 24), v6, v7)) )
      return v10;
    v11 = v3[2];
    v10 = v3 + 2;
    if ( *(_QWORD *)(v11 + 32) == v7 && (!v7 || !memcmp(*(const void **)(v11 + 24), v6, v7)) )
      return v10;
    v12 = v3[3];
    v10 = v3 + 3;
    if ( *(_QWORD *)(v12 + 32) == v7 && (!v7 || !memcmp(*(const void **)(v12 + 24), v6, v7)) )
      return v10;
    v3 += 4;
    if ( v3 == v8 )
    {
      v5 = (a2 - (__int64)v3) >> 3;
      goto LABEL_19;
    }
  }
  return v3;
}
