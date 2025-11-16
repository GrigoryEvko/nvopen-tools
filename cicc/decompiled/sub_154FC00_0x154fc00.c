// Function: sub_154FC00
// Address: 0x154fc00
//
_BYTE *__fastcall sub_154FC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _BYTE *result; // rax
  char v22; // si
  unsigned __int8 *v23; // rcx
  __int64 v24; // [rsp+0h] [rbp-60h] BYREF
  __int64 v25; // [rsp+8h] [rbp-58h]
  char *v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v9 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v9 <= 0xBu )
  {
    sub_16E7EE0(a1, "!DISubrange(", 12);
  }
  else
  {
    qmemcpy(v9, "!DISubrange(", 12);
    *(_QWORD *)(a1 + 24) += 12LL;
  }
  v24 = a1;
  v26 = ", ";
  v10 = *(unsigned int *)(a2 + 8);
  LOBYTE(v25) = 1;
  v27 = a3;
  v11 = *(_QWORD *)(a2 - 8 * v10);
  v28 = a4;
  v29 = a5;
  v12 = *(unsigned __int8 *)v11;
  if ( (_BYTE)v12 != 1 )
  {
    v13 = v11 | 4;
    v22 = 1;
    v23 = 0;
    if ( (unsigned int)(v12 - 24) > 1 )
      goto LABEL_14;
    goto LABEL_12;
  }
  v13 = *(_QWORD *)(v11 + 136);
  if ( (v13 & 4) != 0 )
  {
LABEL_13:
    v23 = (unsigned __int8 *)(v13 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_14;
  }
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v22 = 0;
LABEL_12:
    v23 = 0;
    if ( v22 )
      goto LABEL_13;
LABEL_14:
    sub_154F950((__int64)&v24, "count", 5u, v23, 0);
    goto LABEL_9;
  }
  v15 = *(_DWORD *)(v14 + 32);
  v16 = *(__int64 **)(v14 + 24);
  if ( v15 > 0x40 )
    v17 = *v16;
  else
    v17 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
  sub_154AEF0((__int64)&v24, "count", 5u, v17, 0);
LABEL_9:
  sub_154AEF0((__int64)&v24, "lowerBound", 0xAu, *(_QWORD *)(a2 + 24), 1);
  result = *(_BYTE **)(a1 + 24);
  if ( *(_BYTE **)(a1 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a1, ")", 1, v18, v19, v20, v24, v25, v26, v27, v28, v29);
  *result = 41;
  ++*(_QWORD *)(a1 + 24);
  return result;
}
