// Function: sub_2EF99C0
// Address: 0x2ef99c0
//
__int64 __fastcall sub_2EF99C0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  _DWORD *v4; // rsi
  __int64 v5; // r8
  _DWORD *i; // r11
  int v7; // eax
  __int64 v8; // r10
  int v9; // edx
  unsigned int v10; // eax
  int *v11; // rdi
  int v12; // r9d
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int *v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rcx
  unsigned int *v19; // r12
  unsigned int v20; // r13d
  int v21; // eax
  _DWORD *v22; // rcx
  _DWORD *j; // r11
  int v24; // eax
  __int64 v25; // r10
  int v26; // edx
  unsigned int v27; // eax
  int *v28; // rdi
  int v29; // r9d
  __int64 result; // rax
  int v31; // edi
  int v32; // ebx
  int v33; // edi
  int v34; // ebx
  __int64 v35; // [rsp+0h] [rbp-50h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v36[0] = a2;
  v3 = sub_2EEFC50(a1 + 600, v36);
  sub_2EF97A0((__int64)(v3 + 5), a1 + 464);
  v4 = *(_DWORD **)(a1 + 464);
  v5 = a1 + 272;
  for ( i = &v4[*(unsigned int *)(a1 + 472)]; i != v4; ++v4 )
  {
    v7 = *(_DWORD *)(a1 + 296);
    v8 = *(_QWORD *)(a1 + 280);
    if ( v7 )
    {
      v9 = v7 - 1;
      v10 = (v7 - 1) & (37 * *v4);
      v11 = (int *)(v8 + 4LL * (v9 & (unsigned int)(37 * *v4)));
      v12 = *v11;
      if ( *v11 == *v4 )
      {
LABEL_4:
        *v11 = -2;
        --*(_DWORD *)(a1 + 288);
        ++*(_DWORD *)(a1 + 292);
      }
      else
      {
        v31 = 1;
        while ( v12 != -1 )
        {
          v32 = v31 + 1;
          v10 = v9 & (v31 + v10);
          v11 = (int *)(v8 + 4LL * v10);
          v12 = *v11;
          if ( *v4 == *v11 )
            goto LABEL_4;
          v31 = v32;
        }
      }
    }
  }
  v13 = *(_DWORD *)(a1 + 552);
  v14 = *(unsigned int *)(a1 + 392);
  for ( *(_DWORD *)(a1 + 472) = 0; v13; v13 = *(_DWORD *)(a1 + 552) )
  {
    while ( 1 )
    {
      v15 = v13;
      v16 = *(unsigned int **)(a1 + 280);
      --v13;
      v17 = *(_QWORD *)(*(_QWORD *)(a1 + 544) + 8 * v15 - 8);
      v18 = *(unsigned int *)(a1 + 296);
      *(_DWORD *)(a1 + 552) = v13;
      v19 = &v16[v18];
      if ( *(_DWORD *)(a1 + 288) )
      {
        if ( v16 != v19 )
        {
          while ( *v16 > 0xFFFFFFFD )
          {
            if ( ++v16 == v19 )
              goto LABEL_8;
          }
          if ( v16 != v19 )
            break;
        }
      }
LABEL_8:
      if ( !v13 )
        goto LABEL_21;
    }
LABEL_15:
    v20 = *v16;
    if ( *v16 - 1 <= 0x3FFFFFFE )
    {
      v21 = *(_DWORD *)(v17 + 4LL * (v20 >> 5));
      if ( !_bittest(&v21, v20) )
      {
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 396) )
        {
          v35 = v5;
          sub_C8D5F0(a1 + 384, (const void *)(a1 + 400), v14 + 1, 4u, v5, v14 + 1);
          v14 = *(unsigned int *)(a1 + 392);
          v5 = v35;
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 384) + 4 * v14) = v20;
        v14 = (unsigned int)(*(_DWORD *)(a1 + 392) + 1);
        *(_DWORD *)(a1 + 392) = v14;
      }
    }
    while ( ++v16 != v19 )
    {
      if ( *v16 <= 0xFFFFFFFD )
      {
        if ( v16 != v19 )
          goto LABEL_15;
        break;
      }
    }
  }
LABEL_21:
  v22 = *(_DWORD **)(a1 + 384);
  for ( j = &v22[v14]; j != v22; ++v22 )
  {
    v24 = *(_DWORD *)(a1 + 296);
    v25 = *(_QWORD *)(a1 + 280);
    if ( v24 )
    {
      v26 = v24 - 1;
      v27 = (v24 - 1) & (37 * *v22);
      v28 = (int *)(v25 + 4LL * (v26 & (unsigned int)(37 * *v22)));
      v29 = *v28;
      if ( *v28 == *v22 )
      {
LABEL_24:
        *v28 = -2;
        --*(_DWORD *)(a1 + 288);
        ++*(_DWORD *)(a1 + 292);
      }
      else
      {
        v33 = 1;
        while ( v29 != -1 )
        {
          v34 = v33 + 1;
          v27 = v26 & (v33 + v27);
          v28 = (int *)(v25 + 4LL * v27);
          v29 = *v28;
          if ( *v22 == *v28 )
            goto LABEL_24;
          v33 = v34;
        }
      }
    }
  }
  *(_DWORD *)(a1 + 392) = 0;
  result = sub_2EF97A0(v5, a1 + 304);
  *(_DWORD *)(a1 + 312) = 0;
  return result;
}
