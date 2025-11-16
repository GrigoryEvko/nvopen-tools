// Function: sub_3569CB0
// Address: 0x3569cb0
//
unsigned __int64 __fastcall sub_3569CB0(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int64 result; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r15
  _QWORD *v17; // rax
  char v18; // dl
  unsigned __int64 *v19; // r9
  _QWORD *v20; // rcx
  unsigned __int64 *v21; // rdx
  unsigned __int64 *v22; // r8
  unsigned __int64 v23; // rsi
  _QWORD *v24; // rax
  _QWORD *i; // rsi
  __m128i v26; // [rsp+0h] [rbp-50h] BYREF
  char v27; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 104);
  while ( 1 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    v4 = v3 & 0xFFFFFFFFFFFFFFF9LL;
    if ( *(_BYTE *)(v2 - 8) )
      goto LABEL_3;
    v23 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
    v24 = *(_QWORD **)(v23 + 112);
    v5 = v4 | (*(_QWORD *)v3 >> 1) & 2LL;
    if ( ((*(_QWORD *)v3 >> 1) & 2LL) != 0 )
    {
      if ( *(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 8)
                                                                                        + 32LL) )
        v5 = *(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      for ( i = &v24[*(unsigned int *)(v23 + 120)]; v24 != i; ++v24 )
      {
        if ( *v24 != *(_QWORD *)(*(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) )
          break;
      }
    }
    *(_QWORD *)(v2 - 24) = v5;
    *(_QWORD *)(v2 - 16) = v24;
    *(_BYTE *)(v2 - 8) = 1;
LABEL_4:
    v6 = v4 | 4;
    if ( (*(_QWORD *)v3 & 4) == 0 )
      v6 = v3 & 0xFFFFFFFFFFFFFFF9LL;
    if ( ((v5 >> 1) & 3) == 0 )
      break;
    result = (v6 >> 1) & 3;
    if ( ((v5 >> 1) & 3) != (_DWORD)result )
    {
      v8 = v5;
      v9 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 - 24) = v8 & 0xFFFFFFFFFFFFFFF9LL | 4;
      v10 = *(_QWORD *)(v9 + 32);
      v11 = *(_QWORD **)(v9 + 8);
      goto LABEL_9;
    }
LABEL_23:
    *(_QWORD *)(a1 + 104) -= 32LL;
    v2 = *(_QWORD *)(a1 + 104);
    if ( v2 == *(_QWORD *)(a1 + 96) )
      return result;
  }
  v19 = *(unsigned __int64 **)(v2 - 16);
  result = *(_QWORD *)((*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL) + 112)
         + 8LL * *(unsigned int *)((*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL) + 120);
  if ( v19 == (unsigned __int64 *)result )
    goto LABEL_23;
  v20 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
  v21 = v19 + 1;
  do
  {
    *(_QWORD *)(v2 - 16) = v21;
    v22 = v21;
    v11 = (_QWORD *)v20[1];
    if ( v21 == (unsigned __int64 *)(*(_QWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 112)
                                   + 8LL * *(unsigned int *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 120)) )
      break;
    ++v21;
  }
  while ( *v22 == v11[4] );
  v10 = *v19;
LABEL_9:
  v16 = sub_3569C80(v11, v10);
  if ( !*(_BYTE *)(a1 + 28) )
  {
LABEL_16:
    sub_C8CC70(a1, (__int64)v16, (__int64)v12, v13, v14, v15);
    if ( v18 )
      goto LABEL_15;
    goto LABEL_3;
  }
  v17 = *(_QWORD **)(a1 + 8);
  v13 = *(unsigned int *)(a1 + 20);
  v12 = &v17[v13];
  if ( v17 != v12 )
  {
    while ( v16 != (_QWORD *)*v17 )
    {
      if ( v12 == ++v17 )
        goto LABEL_13;
    }
LABEL_3:
    v5 = *(_QWORD *)(v2 - 24);
    goto LABEL_4;
  }
LABEL_13:
  if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 16) )
    goto LABEL_16;
  *(_DWORD *)(a1 + 20) = v13 + 1;
  *v12 = v16;
  ++*(_QWORD *)a1;
LABEL_15:
  v26.m128i_i64[0] = (__int64)v16;
  v27 = 0;
  return sub_3569710((unsigned __int64 *)(a1 + 96), &v26);
}
