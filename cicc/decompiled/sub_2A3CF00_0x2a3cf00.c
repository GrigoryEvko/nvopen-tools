// Function: sub_2A3CF00
// Address: 0x2a3cf00
//
__int64 __fastcall sub_2A3CF00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  const void *v6; // r13
  size_t v7; // r14
  __int64 v8; // rbx
  size_t v9; // rdx
  size_t v11; // rdx
  __int64 v12; // r15
  size_t v13; // rdx
  size_t v14; // rdx
  const void *v15; // r13
  size_t v16; // rbx
  size_t v17; // rdx
  size_t v18; // rdx
  size_t v19; // rdx

  v3 = a1;
  v4 = (a2 - a1) >> 6;
  v5 = (a2 - a1) >> 4;
  if ( v4 <= 0 )
  {
LABEL_18:
    switch ( v5 )
    {
      case 2LL:
        v15 = *(const void **)a3;
        v16 = *(_QWORD *)(a3 + 8);
        break;
      case 3LL:
        v18 = *(_QWORD *)(v3 + 8);
        v16 = *(_QWORD *)(a3 + 8);
        v15 = *(const void **)a3;
        if ( v16 >= v18 && (!v18 || !memcmp(*(const void **)a3, *(const void **)v3, v18)) )
          return v3;
        v3 += 16;
        break;
      case 1LL:
        v15 = *(const void **)a3;
        v16 = *(_QWORD *)(a3 + 8);
LABEL_24:
        v17 = *(_QWORD *)(v3 + 8);
        if ( v16 >= v17 )
        {
          if ( v17 && memcmp(v15, *(const void **)v3, v17) )
            return a2;
          return v3;
        }
        return a2;
      default:
        return a2;
    }
    v19 = *(_QWORD *)(v3 + 8);
    if ( v16 >= v19 && (!v19 || !memcmp(v15, *(const void **)v3, v19)) )
      return v3;
    v3 += 16;
    goto LABEL_24;
  }
  v6 = *(const void **)a3;
  v7 = *(_QWORD *)(a3 + 8);
  v8 = a1 + (v4 << 6);
  while ( 1 )
  {
    v9 = *(_QWORD *)(v3 + 8);
    if ( v9 <= v7 && (!v9 || !memcmp(v6, *(const void **)v3, v9)) )
      return v3;
    v11 = *(_QWORD *)(v3 + 24);
    v12 = v3 + 16;
    if ( v7 >= v11 && (!v11 || !memcmp(v6, *(const void **)(v3 + 16), v11)) )
      return v12;
    v13 = *(_QWORD *)(v3 + 40);
    v12 = v3 + 32;
    if ( v7 >= v13 && (!v13 || !memcmp(v6, *(const void **)(v3 + 32), v13)) )
      return v12;
    v14 = *(_QWORD *)(v3 + 56);
    v12 = v3 + 48;
    if ( v7 >= v14 && (!v14 || !memcmp(v6, *(const void **)(v3 + 48), v14)) )
      return v12;
    v3 += 64;
    if ( v3 == v8 )
    {
      v5 = (a2 - v3) >> 4;
      goto LABEL_18;
    }
  }
}
