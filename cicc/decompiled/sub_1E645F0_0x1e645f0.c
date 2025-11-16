// Function: sub_1E645F0
// Address: 0x1e645f0
//
__int64 __fastcall sub_1E645F0(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rdi
  _QWORD *v12; // r15
  _QWORD *v13; // rax
  char v14; // dl
  unsigned __int64 *v15; // r8
  _QWORD *v16; // rcx
  _QWORD *v17; // rsi
  _QWORD *v18; // rcx
  unsigned int v19; // edi
  _QWORD *v20; // rsi
  unsigned __int64 v21; // rsi
  _QWORD *v22; // rax
  _QWORD *i; // rsi
  _QWORD *v24; // [rsp+0h] [rbp-50h] BYREF
  char v25; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 112);
  while ( 1 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    v4 = v3 & 0xFFFFFFFFFFFFFFF9LL;
    if ( *(_BYTE *)(v2 - 8) )
      goto LABEL_3;
    v21 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
    v22 = *(_QWORD **)(v21 + 88);
    v5 = v4 | (*(_QWORD *)v3 >> 1) & 2LL;
    if ( ((*(_QWORD *)v3 >> 1) & 2LL) != 0 )
    {
      if ( *(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 8)
                                                                                        + 32LL) )
        v5 = *(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      for ( i = *(_QWORD **)(v21 + 96); v22 != i; ++v22 )
      {
        if ( *v22 != *(_QWORD *)(*(_QWORD *)((*(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) )
          break;
      }
    }
    *(_BYTE *)(v2 - 8) = 1;
    *(_QWORD *)(v2 - 24) = v5;
    *(_QWORD *)(v2 - 16) = v22;
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
LABEL_28:
    *(_QWORD *)(a1 + 112) -= 32LL;
    v2 = *(_QWORD *)(a1 + 112);
    if ( v2 == *(_QWORD *)(a1 + 104) )
      return result;
  }
  v15 = *(unsigned __int64 **)(v2 - 16);
  v16 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
  result = (__int64)(v15 + 1);
  if ( v15 == *(unsigned __int64 **)((*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL) + 96) )
    goto LABEL_28;
  do
  {
    *(_QWORD *)(v2 - 16) = result;
    v17 = (_QWORD *)result;
    v11 = (_QWORD *)v16[1];
    if ( result == *(_QWORD *)((*v16 & 0xFFFFFFFFFFFFFFF8LL) + 96) )
      break;
    result += 8;
  }
  while ( *v17 == v11[4] );
  v10 = *v15;
LABEL_9:
  v12 = sub_1E64130(v11, v10);
  v13 = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)(a1 + 16) != v13 )
  {
LABEL_10:
    sub_16CCBA0(a1, (__int64)v12);
    if ( v14 )
      goto LABEL_11;
    goto LABEL_3;
  }
  v18 = &v13[*(unsigned int *)(a1 + 28)];
  v19 = *(_DWORD *)(a1 + 28);
  if ( v13 != v18 )
  {
    v20 = 0;
    while ( v12 != (_QWORD *)*v13 )
    {
      if ( *v13 == -2 )
      {
        v20 = v13;
        if ( v13 + 1 == v18 )
          goto LABEL_23;
        ++v13;
      }
      else if ( v18 == ++v13 )
      {
        if ( !v20 )
          goto LABEL_26;
LABEL_23:
        *v20 = v12;
        --*(_DWORD *)(a1 + 32);
        ++*(_QWORD *)a1;
        goto LABEL_11;
      }
    }
LABEL_3:
    v5 = *(_QWORD *)(v2 - 24);
    goto LABEL_4;
  }
LABEL_26:
  if ( v19 >= *(_DWORD *)(a1 + 24) )
    goto LABEL_10;
  *(_DWORD *)(a1 + 28) = v19 + 1;
  *v18 = v12;
  ++*(_QWORD *)a1;
LABEL_11:
  v24 = v12;
  v25 = 0;
  return sub_1E645A0((__int64 *)(a1 + 104), (__int64)&v24);
}
