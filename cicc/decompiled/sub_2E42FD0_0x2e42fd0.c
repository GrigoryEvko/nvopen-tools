// Function: sub_2E42FD0
// Address: 0x2e42fd0
//
__int64 __fastcall sub_2E42FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  int v13; // ecx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rsi
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // r12d
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *j; // rdx
  __int64 v26; // rax
  int v27; // edi
  _QWORD *v28; // rax
  _BYTE v29[64]; // [rsp+0h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 112) = a3;
  *(_QWORD *)(a1 + 120) = a4;
  *(_QWORD *)(a1 + 128) = a2;
  sub_FE9370(a1);
  v5 = *(_QWORD *)(a1 + 136);
  if ( *(_QWORD *)(a1 + 144) != v5 )
    *(_QWORD *)(a1 + 144) = v5;
  v6 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 180) )
      goto LABEL_9;
    v7 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 168), 16LL * (unsigned int)v7, 8);
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v17 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 184);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v17 = 64;
  if ( (unsigned int)v7 <= v17 )
  {
LABEL_6:
    v8 = *(_QWORD **)(a1 + 168);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 176) = 0;
    goto LABEL_9;
  }
  v18 = v6 - 1;
  if ( !v18 )
  {
    v19 = *(_QWORD **)(a1 + 168);
    v20 = 64;
LABEL_25:
    sub_C7D6A0((__int64)v19, 16LL * (unsigned int)v7, 8);
    v21 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
    v22 = (v21
         | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 184) = v22;
    v23 = (_QWORD *)sub_C7D670(16 * v22, 8);
    v24 = *(unsigned int *)(a1 + 184);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = v23;
    for ( j = &v23[2 * v24]; j != v23; v23 += 2 )
    {
      if ( v23 )
        *v23 = -4096;
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v18, v18);
  v19 = *(_QWORD **)(a1 + 168);
  v20 = 1 << (33 - (v18 ^ 0x1F));
  if ( v20 < 64 )
    v20 = 64;
  if ( (_DWORD)v7 != v20 )
    goto LABEL_25;
  *(_QWORD *)(a1 + 176) = 0;
  v28 = &v19[2 * (unsigned int)v7];
  do
  {
    if ( v19 )
      *v19 = -4096;
    v19 += 2;
  }
  while ( v28 != v19 );
LABEL_9:
  sub_2E41F60(a1);
  sub_2E3C430(a1);
  sub_2E42F50(a1);
  if ( !(unsigned __int8)sub_2E3AC80((_QWORD *)a1) )
  {
    sub_2E42D20(a1, 0, *(_QWORD *)(a1 + 88));
    if ( !(unsigned __int8)sub_2E3AC80((_QWORD *)a1) )
      BUG();
  }
  sub_FE9700((_QWORD *)a1);
  if ( LOBYTE(qword_4F8E448[8]) )
  {
    sub_B2EE70((__int64)v29, **(_QWORD **)(a1 + 128), 0);
    if ( v29[16] )
    {
      v26 = a1 + 88;
      while ( *(_QWORD *)(a1 + 88) != v26 )
      {
        v26 = *(_QWORD *)(v26 + 8);
        if ( *(_DWORD *)(v26 + 28) > 1u )
        {
          sub_2E404A0(a1);
          break;
        }
      }
    }
  }
  sub_FE98F0((_QWORD *)a1);
  result = (__int64)&qword_4F8E4E0;
  if ( LOBYTE(qword_4F8E528[8]) )
  {
    v11 = *(_QWORD *)(a2 + 328);
    v12 = a2 + 320;
    if ( a2 + 320 != v11 )
    {
      while ( 1 )
      {
        v15 = *(_DWORD *)(a1 + 184);
        v16 = *(_QWORD *)(a1 + 168);
        if ( !v15 )
          goto LABEL_17;
        v13 = v15 - 1;
        result = (v15 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v14 = *(_QWORD *)(v16 + 16 * result);
        if ( v14 != v11 )
          break;
LABEL_15:
        v11 = *(_QWORD *)(v11 + 8);
        if ( v12 == v11 )
          return result;
      }
      v27 = 1;
      while ( v14 != -4096 )
      {
        result = v13 & (unsigned int)(v27 + result);
        v14 = *(_QWORD *)(v16 + 16LL * (unsigned int)result);
        if ( v14 == v11 )
          goto LABEL_15;
        ++v27;
      }
LABEL_17:
      result = sub_2E3D9B0(a1, v11, 0);
      goto LABEL_15;
    }
  }
  return result;
}
