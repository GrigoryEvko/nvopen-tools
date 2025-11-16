// Function: sub_2BDD0F0
// Address: 0x2bdd0f0
//
__int64 __fastcall sub_2BDD0F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rax
  const void *v7; // r13
  size_t v8; // r15
  const void *v10; // r13
  size_t v11; // r12

  v4 = a1;
  v5 = a2 - a1;
  v6 = v5 >> 5;
  if ( v5 >> 7 > 0 )
  {
    v7 = *(const void **)a3;
    v8 = *(_QWORD *)(a3 + 8);
    while ( *(_QWORD *)(v4 + 8) != v8 || v8 && memcmp(*(const void **)v4, v7, v8) )
    {
      if ( v8 == *(_QWORD *)(v4 + 40) && (!v8 || !memcmp(*(const void **)(v4 + 32), v7, v8)) )
        return v4 + 32;
      if ( v8 == *(_QWORD *)(v4 + 72) && (!v8 || !memcmp(*(const void **)(v4 + 64), v7, v8)) )
        return v4 + 64;
      if ( v8 == *(_QWORD *)(v4 + 104) && (!v8 || !memcmp(*(const void **)(v4 + 96), v7, v8)) )
        return v4 + 96;
      v4 += 128;
      if ( v4 == a1 + (v5 >> 7 << 7) )
      {
        v6 = (a2 - v4) >> 5;
        goto LABEL_21;
      }
    }
    return v4;
  }
LABEL_21:
  switch ( v6 )
  {
    case 2LL:
      v10 = *(const void **)a3;
      v11 = *(_QWORD *)(a3 + 8);
LABEL_33:
      if ( *(_QWORD *)(v4 + 8) == v11 && (!v11 || !memcmp(*(const void **)v4, v10, v11)) )
        return v4;
      v4 += 32;
LABEL_28:
      if ( *(_QWORD *)(v4 + 8) != v11 || v11 && memcmp(*(const void **)v4, v10, v11) )
        return a2;
      return v4;
    case 3LL:
      v10 = *(const void **)a3;
      v11 = *(_QWORD *)(a3 + 8);
      if ( *(_QWORD *)(v4 + 8) == v11 && (!v11 || !memcmp(*(const void **)v4, *(const void **)a3, *(_QWORD *)(a3 + 8))) )
        return v4;
      v4 += 32;
      goto LABEL_33;
    case 1LL:
      v10 = *(const void **)a3;
      v11 = *(_QWORD *)(a3 + 8);
      goto LABEL_28;
  }
  return a2;
}
