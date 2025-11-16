// Function: sub_77C420
// Address: 0x77c420
//
__int64 __fastcall sub_77C420(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 *v9; // rcx
  __int64 v10; // rax
  char i; // dl
  FILE *v14; // r14
  _QWORD *v15; // r15
  __int64 v16; // r12
  __int64 v17; // rax
  char j; // r8
  __int64 v19; // rdx
  char k; // cl
  __int64 result; // rax
  unsigned __int64 v22; // rax
  int v23; // r9d
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  char m; // al
  __int64 v27; // rcx
  __int64 v28; // rdx
  unsigned int v29; // ecx
  _QWORD *v30; // r15
  int v31; // edx
  unsigned int v32; // eax
  unsigned int v33; // [rsp+8h] [rbp-48h]
  unsigned int v34; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v35[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = a3[9];
  v34 = 1;
  if ( !v4 )
    return 1;
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 1;
  v6 = *(__int64 **)(v5 + 16);
  if ( !v6 )
    return 1;
  v9 = (__int64 *)v6[2];
  if ( !v9 )
    return 1;
  v10 = *v6;
  for ( i = *(_BYTE *)(v10 + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
    v10 = *(_QWORD *)(v10 + 160);
  v14 = (FILE *)((char *)a3 + 28);
  if ( i != 6 )
    goto LABEL_18;
  v15 = *(_QWORD **)(a4 + 8);
  v16 = *(_QWORD *)(a1 + 184);
  if ( !v16 )
  {
LABEL_43:
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xBAFu, (FILE *)((char *)a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  while ( *v15 != v16 + *(unsigned int *)(v16 + 40) )
  {
    v16 = *(_QWORD *)v16;
    if ( !v16 )
      goto LABEL_43;
  }
  v17 = *v9;
  for ( j = *(_BYTE *)(*v9 + 140); j == 12; j = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  v19 = *a3;
  for ( k = *(_BYTE *)(*a3 + 140); k == 12; k = *(_BYTE *)(v19 + 140) )
    v19 = *(_QWORD *)(v19 + 160);
  if ( k != 1 || j != 2 )
  {
LABEL_18:
    v34 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_687670(0xBB4u, (__int64)v14, *a2, a2[19], (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v34;
    }
    return 0;
  }
  v22 = sub_620EE0(*(_WORD **)(a4 + 16), byte_4B6DF90[*(unsigned __int8 *)(v17 + 160)], v35);
  v23 = v22;
  if ( !v35[0] && v22 <= 0xFFFFFF )
  {
    v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2[5] + 32) + 168LL) + 168LL);
    if ( !v24 || *(_BYTE *)(v24 + 8) )
      goto LABEL_18;
    v25 = *(_QWORD *)(v24 + 32);
    for ( m = *(_BYTE *)(v25 + 140); m == 12; m = *(_BYTE *)(v25 + 140) )
      v25 = *(_QWORD *)(v25 + 160);
    v27 = *(_QWORD *)(v16 + 16);
    if ( v27 != v25 )
    {
      if ( !v27 || !dword_4F07588 || (v28 = *(_QWORD *)(v27 + 32), *(_QWORD *)(v25 + 32) != v28) || !v28 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
          return 0;
        v30 = (_QWORD *)(a1 + 96);
        sub_687430(0xBB2u, (__int64)v14, v25, v27, (_QWORD *)(a1 + 96));
LABEL_41:
        sub_770D30(a1);
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_6855B0(0xBB1u, (FILE *)(v16 + 24), v30);
          sub_770D30(a1);
          return 0;
        }
        return 0;
      }
    }
    if ( m == 2 || m == 3 )
    {
      v29 = 16;
    }
    else
    {
      v33 = v23;
      v29 = sub_7764B0(a1, v25, &v34);
      result = v34;
      if ( !v34 )
        return result;
      v23 = v33;
      if ( !v29 )
      {
        if ( *(_DWORD *)(v16 + 36) != *(_DWORD *)(v16 + 40) )
        {
LABEL_39:
          if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
            return 0;
          v30 = (_QWORD *)(a1 + 96);
          sub_67E4F0(0xBB0u, v14, v23, v29, (_QWORD *)(a1 + 96));
          goto LABEL_41;
        }
LABEL_53:
        if ( (*(_BYTE *)(v15[3] - 9LL) & 4) != 0 )
        {
          sub_771420(a1, v16);
          return v34;
        }
        else
        {
          sub_770DD0(0xCF0u, v14, a1);
          sub_770DD0(0xBB1u, (FILE *)(v16 + 24), a1);
          return 0;
        }
      }
      if ( 0x10000000 / v29 < v33 )
      {
        sub_770DD0(0xBAEu, v14, a1);
        return 0;
      }
    }
    v31 = v23 * v29;
    if ( (((_BYTE)v23 * (_BYTE)v29) & 7) != 0 )
      v31 = v23 * v29 + 8 - (((_BYTE)v23 * (_BYTE)v29) & 7);
    v32 = *(_DWORD *)(v16 + 36) - *(_DWORD *)(v16 + 40);
    if ( v32 != v31 )
    {
      v29 = v32 / v29;
      goto LABEL_39;
    }
    goto LABEL_53;
  }
  if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
    return 0;
  sub_67E440(0xBADu, v14, v22, (_QWORD *)(a1 + 96));
  sub_770D30(a1);
  return 0;
}
