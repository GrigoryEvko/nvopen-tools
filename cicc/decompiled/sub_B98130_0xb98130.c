// Function: sub_B98130
// Address: 0xb98130
//
void __fastcall sub_B98130(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64, __int64, _QWORD), __int64 a3)
{
  __int64 v5; // rax
  unsigned int v6; // edx
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rcx
  unsigned int *v10; // r14
  __int64 v11; // rax
  unsigned int *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int *v15; // r15
  unsigned int *i; // r15
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int *v20; // r13
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rcx
  int v27; // r8d
  __int64 v28; // [rsp+18h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return;
  v5 = *(_QWORD *)sub_BD5C60(a1, a2);
  v6 = *(_DWORD *)(v5 + 3248);
  v7 = *(_QWORD *)(v5 + 3232);
  if ( v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v28 = v7 + 40LL * v8;
    v9 = *(_QWORD *)v28;
    if ( a1 == *(_QWORD *)v28 )
      goto LABEL_5;
    v27 = 1;
    while ( v9 != -4096 )
    {
      v8 = (v6 - 1) & (v27 + v8);
      v28 = v7 + 40LL * v8;
      v9 = *(_QWORD *)v28;
      if ( a1 == *(_QWORD *)v28 )
        goto LABEL_5;
      ++v27;
    }
  }
  v28 = v7 + 40LL * v6;
LABEL_5:
  v10 = *(unsigned int **)(v28 + 8);
  v11 = 16LL * *(unsigned int *)(v28 + 16);
  v12 = &v10[(unsigned __int64)v11 / 4];
  v13 = v11 >> 4;
  v14 = v11 >> 6;
  if ( v14 )
  {
    v15 = &v10[16 * v14];
    while ( 1 )
    {
      v7 = *v10;
      if ( a2(a3, v7, *((_QWORD *)v10 + 1)) )
        goto LABEL_12;
      v7 = v10[4];
      if ( a2(a3, v7, *((_QWORD *)v10 + 3)) )
      {
        v10 += 4;
        goto LABEL_12;
      }
      v7 = v10[8];
      if ( a2(a3, v7, *((_QWORD *)v10 + 5)) )
      {
        v10 += 8;
        goto LABEL_12;
      }
      v7 = v10[12];
      if ( a2(a3, v7, *((_QWORD *)v10 + 7)) )
      {
        v10 += 12;
        goto LABEL_12;
      }
      v10 += 16;
      if ( v15 == v10 )
      {
        v13 = ((char *)v12 - (char *)v10) >> 4;
        break;
      }
    }
  }
  switch ( v13 )
  {
    case 2LL:
LABEL_53:
      v7 = *v10;
      if ( a2(a3, v7, *((_QWORD *)v10 + 1)) )
        goto LABEL_12;
      v10 += 4;
      goto LABEL_55;
    case 3LL:
      v7 = *v10;
      if ( a2(a3, v7, *((_QWORD *)v10 + 1)) )
        goto LABEL_12;
      v10 += 4;
      goto LABEL_53;
    case 1LL:
LABEL_55:
      v7 = *v10;
      if ( !a2(a3, v7, *((_QWORD *)v10 + 1)) )
        break;
LABEL_12:
      if ( v12 != v10 && v12 != v10 + 4 )
      {
        for ( i = v10 + 6; ; i += 4 )
        {
          v7 = *(i - 2);
          if ( !a2(a3, v7, *(_QWORD *)i) )
          {
            v17 = (__int64)(v10 + 2);
            *v10 = *(i - 2);
            if ( v10 + 2 != i )
            {
              v18 = *((_QWORD *)v10 + 1);
              if ( v18 )
              {
                sub_B91220((__int64)(v10 + 2), v18);
                v17 = (__int64)(v10 + 2);
              }
              v7 = *(_QWORD *)i;
              *((_QWORD *)v10 + 1) = *(_QWORD *)i;
              if ( v7 )
              {
                sub_B976B0((__int64)i, (unsigned __int8 *)v7, v17);
                *(_QWORD *)i = 0;
              }
            }
            v10 += 4;
          }
          if ( v12 == i + 2 )
            break;
        }
      }
      goto LABEL_24;
  }
  v10 = v12;
LABEL_24:
  v19 = *(_QWORD *)(v28 + 8);
  v20 = (unsigned int *)(v19 + 16LL * *(unsigned int *)(v28 + 16));
  v21 = (char *)v20 - (char *)v12;
  v22 = ((char *)v20 - (char *)v12) >> 4;
  if ( (char *)v20 - (char *)v12 > 0 )
  {
    v23 = (__int64 *)(v12 + 2);
    v24 = (__int64 *)(v10 + 2);
    do
    {
      *((_DWORD *)v24 - 2) = *((_DWORD *)v23 - 2);
      if ( v24 != v23 )
      {
        if ( *v24 )
          sub_B91220((__int64)v24, *v24);
        v7 = *v23;
        *v24 = *v23;
        if ( v7 )
        {
          sub_B976B0((__int64)v23, (unsigned __int8 *)v7, (__int64)v24);
          *v23 = 0;
        }
      }
      v23 += 2;
      v24 += 2;
      --v22;
    }
    while ( v22 );
    v25 = 16;
    if ( v21 > 0 )
      v25 = v21;
    v10 = (unsigned int *)((char *)v10 + v25);
    v19 = *(_QWORD *)(v28 + 8);
    v20 = (unsigned int *)(v19 + 16LL * *(unsigned int *)(v28 + 16));
  }
  if ( v10 != v20 )
  {
    do
    {
      v7 = *((_QWORD *)v20 - 1);
      v20 -= 4;
      if ( v7 )
        sub_B91220((__int64)(v20 + 2), v7);
    }
    while ( v20 != v10 );
    v19 = *(_QWORD *)(v28 + 8);
  }
  v26 = ((__int64)v10 - v19) >> 4;
  *(_DWORD *)(v28 + 16) = v26;
  if ( !(_DWORD)v26 )
    sub_B91E30(a1, v7);
}
