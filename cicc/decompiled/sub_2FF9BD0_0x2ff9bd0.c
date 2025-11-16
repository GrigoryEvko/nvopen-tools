// Function: sub_2FF9BD0
// Address: 0x2ff9bd0
//
char __fastcall sub_2FF9BD0(_QWORD *a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // eax
  int v9; // ecx
  unsigned int v11; // eax
  __int64 v12; // rdx
  char result; // al
  int v14; // edi
  unsigned __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned int v19; // r14d
  __int16 *v20; // r15
  __int64 v21; // r12
  int v22; // edx
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // rax
  __int64 *v33; // rdx
  _QWORD *v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  char v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  v4 = a3;
  v5 = a1[6];
  if ( !v5 )
    return (unsigned int)sub_2E89C70(a2, v4, 0, 1) != -1;
  v6 = *(_QWORD *)(v5 + 32);
  v7 = *(_QWORD *)(v6 + 128);
  v8 = *(_DWORD *)(v6 + 144);
  if ( !v8 )
    return (unsigned int)sub_2E89C70(a2, v4, 0, 1) != -1;
  v9 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = *(_QWORD *)(v7 + 16LL * v11);
  if ( v12 != a2 )
  {
    v14 = 1;
    while ( v12 != -4096 )
    {
      v11 = v9 & (v14 + v11);
      v12 = *(_QWORD *)(v7 + 16LL * v11);
      if ( v12 == a2 )
        goto LABEL_4;
      ++v14;
    }
    return (unsigned int)sub_2E89C70(a2, v4, 0, 1) != -1;
  }
LABEL_4:
  if ( (int)v4 < 0 )
  {
    v15 = *(unsigned int *)(v5 + 160);
    v16 = v4 & 0x7FFFFFFF;
    if ( ((unsigned int)v4 & 0x7FFFFFFF) < (unsigned int)v15 )
    {
      v17 = *(_QWORD *)(*(_QWORD *)(v5 + 152) + 8LL * v16);
      if ( v17 )
        goto LABEL_13;
    }
    v23 = v16 + 1;
    if ( (unsigned int)v15 < v23 && v23 != v15 )
    {
      if ( v23 >= v15 )
      {
        v30 = *(_QWORD *)(v5 + 168);
        v31 = v23 - v15;
        if ( v23 > (unsigned __int64)*(unsigned int *)(v5 + 164) )
        {
          v35 = v23 - v15;
          v38 = *(_QWORD *)(v5 + 168);
          sub_C8D5F0(v5 + 152, (const void *)(v5 + 168), v23, 8u, v30, v31);
          v15 = *(unsigned int *)(v5 + 160);
          v31 = v35;
          v30 = v38;
        }
        v24 = *(_QWORD *)(v5 + 152);
        v32 = (__int64 *)(v24 + 8 * v15);
        v33 = &v32[v31];
        if ( v32 != v33 )
        {
          do
            *v32++ = v30;
          while ( v33 != v32 );
          v24 = *(_QWORD *)(v5 + 152);
        }
        *(_DWORD *)(v5 + 160) += v31;
        goto LABEL_24;
      }
      *(_DWORD *)(v5 + 160) = v23;
    }
    v24 = *(_QWORD *)(v5 + 152);
LABEL_24:
    v25 = sub_2E10F30(v4);
    *(_QWORD *)(v24 + 8 * (v4 & 0x7FFFFFFF)) = v25;
    v36 = v25;
    sub_2E11E80((_QWORD *)v5, v25);
    v17 = v36;
LABEL_13:
    if ( !*(_DWORD *)(v17 + 72) )
      return 0;
    return sub_2FF9AB0(a1[6], a2, (__int64 *)v17);
  }
  if ( (*(_QWORD *)(*(_QWORD *)(a1[4] + 384LL) + 8LL * ((unsigned int)v4 >> 6)) & (1LL << v4)) != 0 )
    return 0;
  v18 = a1[2];
  v19 = *(_DWORD *)(*(_QWORD *)(v18 + 8) + 24 * v4 + 16) & 0xFFF;
  v20 = (__int16 *)(*(_QWORD *)(v18 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v18 + 8) + 24 * v4 + 16) >> 12));
  do
  {
    if ( !v20 )
      break;
    v21 = *(_QWORD *)(*(_QWORD *)(a1[6] + 424LL) + 8LL * v19);
    if ( !v21 )
    {
      v34 = (_QWORD *)a1[6];
      v37 = qword_501EA48[8];
      v26 = (_QWORD *)sub_22077B0(0x68u);
      v27 = v34;
      v28 = v19;
      v21 = (__int64)v26;
      if ( v26 )
      {
        *v26 = v26 + 2;
        v26[1] = 0x200000000LL;
        v26[8] = v26 + 10;
        v26[9] = 0x200000000LL;
        if ( v37 )
        {
          v29 = sub_22077B0(0x30u);
          v27 = v34;
          v28 = v19;
          if ( v29 )
          {
            *(_DWORD *)(v29 + 8) = 0;
            *(_QWORD *)(v29 + 16) = 0;
            *(_QWORD *)(v29 + 24) = v29 + 8;
            *(_QWORD *)(v29 + 32) = v29 + 8;
            *(_QWORD *)(v29 + 40) = 0;
          }
          *(_QWORD *)(v21 + 96) = v29;
        }
        else
        {
          v26[12] = 0;
        }
      }
      *(_QWORD *)(v27[53] + 8 * v28) = v21;
      sub_2E11710(v27, v21, v19);
    }
    if ( !*(_DWORD *)(v21 + 72) )
      return 0;
    result = sub_2FF9AB0(a1[6], a2, (__int64 *)v21);
    if ( !result )
      return result;
    v22 = *v20++;
    v19 += v22;
  }
  while ( (_WORD)v22 );
  return 1;
}
