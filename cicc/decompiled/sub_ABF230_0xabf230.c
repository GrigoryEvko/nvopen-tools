// Function: sub_ABF230
// Address: 0xabf230
//
_BYTE *__fastcall sub_ABF230(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  _BYTE *result; // rax
  __int64 *i; // rbx
  _WORD *v6; // rdx
  __int64 v7; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-48h]
  __int64 v9; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-38h]

  v2 = *a1;
  v3 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 == (__int64 *)v3 )
    return result;
  v8 = *((_DWORD *)v2 + 2);
  if ( v8 > 0x40 )
  {
    sub_C43780(&v7, v2);
    v10 = *((_DWORD *)v2 + 6);
    if ( v10 <= 0x40 )
      goto LABEL_4;
LABEL_29:
    sub_C43780(&v9, v2 + 2);
    goto LABEL_5;
  }
  v7 = *v2;
  v10 = *((_DWORD *)v2 + 6);
  if ( v10 > 0x40 )
    goto LABEL_29;
LABEL_4:
  v9 = v2[2];
LABEL_5:
  result = sub_ABEDB0(a2, (__int64)&v7);
  if ( v10 > 0x40 && v9 )
    result = (_BYTE *)j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    result = (_BYTE *)j_j___libc_free_0_0(v7);
  for ( i = v2 + 4; (__int64 *)v3 != i; i += 4 )
  {
    v6 = *(_WORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 > 1u )
    {
      *v6 = 8236;
      *(_QWORD *)(a2 + 32) += 2LL;
      v8 = *((_DWORD *)i + 2);
      if ( v8 <= 0x40 )
        goto LABEL_14;
    }
    else
    {
      sub_CB6200(a2, ", ", 2);
      v8 = *((_DWORD *)i + 2);
      if ( v8 <= 0x40 )
      {
LABEL_14:
        v7 = *i;
        v10 = *((_DWORD *)i + 6);
        if ( v10 <= 0x40 )
          goto LABEL_15;
        goto LABEL_26;
      }
    }
    sub_C43780(&v7, i);
    v10 = *((_DWORD *)i + 6);
    if ( v10 <= 0x40 )
    {
LABEL_15:
      v9 = i[2];
      goto LABEL_16;
    }
LABEL_26:
    sub_C43780(&v9, i + 2);
LABEL_16:
    result = sub_ABEDB0(a2, (__int64)&v7);
    if ( v10 > 0x40 && v9 )
      result = (_BYTE *)j_j___libc_free_0_0(v9);
    if ( v8 > 0x40 )
    {
      if ( v7 )
        result = (_BYTE *)j_j___libc_free_0_0(v7);
    }
  }
  return result;
}
