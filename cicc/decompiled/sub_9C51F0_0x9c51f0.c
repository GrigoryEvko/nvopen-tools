// Function: sub_9C51F0
// Address: 0x9c51f0
//
__int64 __fastcall sub_9C51F0(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int64 v3; // rdi
  char *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  char *v8; // r13
  __int64 v9; // rax
  char *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  char *v14; // [rsp+8h] [rbp-38h]

  v3 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *(char **)(v3 + 24);
  v14 = *(char **)(v3 + 32);
  v6 = (v14 - v5) >> 5;
  v7 = (v14 - v5) >> 3;
  if ( v6 > 0 )
  {
    v8 = &v5[32 * v6];
    while ( *(_QWORD *)(*(_QWORD *)v5 + 32LL) != a3 || a3 && memcmp(*(const void **)(*(_QWORD *)v5 + 24LL), a2, a3) )
    {
      v9 = *((_QWORD *)v5 + 1);
      v10 = v5 + 8;
      if ( *(_QWORD *)(v9 + 32) == a3 && (!a3 || !memcmp(*(const void **)(v9 + 24), a2, a3))
        || (v11 = *((_QWORD *)v5 + 2), v10 = v5 + 16, *(_QWORD *)(v11 + 32) == a3)
        && (!a3 || !memcmp(*(const void **)(v11 + 24), a2, a3))
        || (v12 = *((_QWORD *)v5 + 3), v10 = v5 + 24, *(_QWORD *)(v12 + 32) == a3)
        && (!a3 || !memcmp(*(const void **)(v12 + 24), a2, a3)) )
      {
        v5 = v10;
        break;
      }
      v5 += 32;
      if ( v8 == v5 )
      {
        v7 = (v14 - v5) >> 3;
        goto LABEL_20;
      }
    }
LABEL_16:
    if ( v14 != v5 )
      return *(_QWORD *)v5;
    return 0;
  }
LABEL_20:
  if ( v7 == 2 )
    goto LABEL_26;
  if ( v7 == 3 )
  {
    if ( *(_QWORD *)(*(_QWORD *)v5 + 32LL) == a3 && (!a3 || !memcmp(*(const void **)(*(_QWORD *)v5 + 24LL), a2, a3)) )
      goto LABEL_16;
    v5 += 8;
LABEL_26:
    if ( *(_QWORD *)(*(_QWORD *)v5 + 32LL) == a3 && (!a3 || !memcmp(*(const void **)(*(_QWORD *)v5 + 24LL), a2, a3)) )
      goto LABEL_16;
    v5 += 8;
    goto LABEL_28;
  }
  if ( v7 != 1 )
    return 0;
LABEL_28:
  if ( *(_QWORD *)(*(_QWORD *)v5 + 32LL) != a3 )
    return 0;
  if ( !a3 || !memcmp(*(const void **)(*(_QWORD *)v5 + 24LL), a2, a3) )
    goto LABEL_16;
  return 0;
}
