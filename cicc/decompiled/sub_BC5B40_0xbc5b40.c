// Function: sub_BC5B40
// Address: 0xbc5b40
//
__int64 __fastcall sub_BC5B40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  const void *v6; // r13
  size_t v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r15
  const void *v11; // r15
  size_t v12; // r13

  v3 = a1;
  v4 = (a2 - a1) >> 7;
  v5 = (a2 - a1) >> 5;
  if ( v4 <= 0 )
  {
LABEL_19:
    switch ( v5 )
    {
      case 2LL:
        v11 = *(const void **)a3;
        v12 = *(_QWORD *)(a3 + 8);
        break;
      case 3LL:
        v11 = *(const void **)a3;
        v12 = *(_QWORD *)(a3 + 8);
        if ( *(_QWORD *)(v3 + 8) == v12
          && (!v12 || !memcmp(*(const void **)v3, *(const void **)a3, *(_QWORD *)(a3 + 8))) )
        {
          return v3;
        }
        v3 += 32;
        break;
      case 1LL:
        v11 = *(const void **)a3;
        v12 = *(_QWORD *)(a3 + 8);
LABEL_28:
        if ( *(_QWORD *)(v3 + 8) == v12 )
        {
          if ( v12 && memcmp(*(const void **)v3, v11, v12) )
            return a2;
          return v3;
        }
        return a2;
      default:
        return a2;
    }
    if ( *(_QWORD *)(v3 + 8) == v12 && (!v12 || !memcmp(*(const void **)v3, v11, v12)) )
      return v3;
    v3 += 32;
    goto LABEL_28;
  }
  v6 = *(const void **)a3;
  v7 = *(_QWORD *)(a3 + 8);
  v8 = a1 + (v4 << 7);
  while ( *(_QWORD *)(v3 + 8) != v7 || v7 && memcmp(*(const void **)v3, v6, v7) )
  {
    v9 = v3 + 32;
    if ( *(_QWORD *)(v3 + 40) == v7 && (!v7 || !memcmp(*(const void **)(v3 + 32), v6, v7)) )
      return v9;
    v9 = v3 + 64;
    if ( *(_QWORD *)(v3 + 72) == v7 && (!v7 || !memcmp(*(const void **)(v3 + 64), v6, v7)) )
      return v9;
    v9 = v3 + 96;
    if ( *(_QWORD *)(v3 + 104) == v7 && (!v7 || !memcmp(*(const void **)(v3 + 96), v6, v7)) )
      return v9;
    v3 += 128;
    if ( v8 == v3 )
    {
      v5 = (a2 - v3) >> 5;
      goto LABEL_19;
    }
  }
  return v3;
}
