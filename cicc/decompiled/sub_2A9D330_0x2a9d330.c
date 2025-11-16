// Function: sub_2A9D330
// Address: 0x2a9d330
//
__int64 __fastcall sub_2A9D330(__int64 **a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // r8
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r14
  int v12; // esi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r8
  __int64 *v18; // r14
  __int64 *v19; // r12
  __int64 v20; // r13
  __int64 v21; // rdi
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // r8d
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r13
  int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-38h]

  result = sub_AA5930(a2);
  v35 = v4;
  if ( result == v4 )
    return result;
  v5 = result;
  do
  {
    v6 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
    v7 = 4LL * v6;
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
    {
      v8 = *(_QWORD **)(v5 - 8);
      v9 = &v8[v7];
      if ( v8 != &v8[v7] )
        goto LABEL_5;
    }
    else
    {
      v9 = (_QWORD *)v5;
      v8 = (_QWORD *)(v5 - v7 * 8);
      if ( v5 - v7 * 8 != v5 )
      {
LABEL_5:
        v10 = **a1;
        do
        {
          if ( v10 == *v8 )
          {
            v11 = *a1[1];
            if ( v6 == *(_DWORD *)(v5 + 72) )
            {
              sub_B48D90(v5);
              v6 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
            }
            v12 = (v6 + 1) & 0x7FFFFFF;
            *(_DWORD *)(v5 + 4) = v12 | *(_DWORD *)(v5 + 4) & 0xF8000000;
            v13 = *(_QWORD *)(v5 - 8) + 32LL * (unsigned int)(v12 - 1);
            if ( *(_QWORD *)v13 )
            {
              v14 = *(_QWORD *)(v13 + 8);
              **(_QWORD **)(v13 + 16) = v14;
              if ( v14 )
                *(_QWORD *)(v14 + 16) = *(_QWORD *)(v13 + 16);
            }
            *(_QWORD *)v13 = v10;
            if ( v10 )
            {
              v15 = *(_QWORD *)(v10 + 16);
              *(_QWORD *)(v13 + 8) = v15;
              if ( v15 )
                *(_QWORD *)(v15 + 16) = v13 + 8;
              *(_QWORD *)(v13 + 16) = v10 + 16;
              *(_QWORD *)(v10 + 16) = v13;
            }
            *(_QWORD *)(*(_QWORD *)(v5 - 8)
                      + 32LL * *(unsigned int *)(v5 + 72)
                      + 8LL * ((*(_DWORD *)(v5 + 4) & 0x7FFFFFFu) - 1)) = v11;
            goto LABEL_18;
          }
          v8 += 4;
        }
        while ( v9 != v8 );
      }
    }
    v16 = 32LL * *(unsigned int *)(v5 + 72);
    v17 = v16 + 8LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
    v18 = (__int64 *)(*(_QWORD *)(v5 - 8) + v16);
    v19 = (__int64 *)(*(_QWORD *)(v5 - 8) + v17);
    if ( v19 == v18 )
      goto LABEL_18;
    while ( 1 )
    {
      v20 = *v18;
      v21 = a1[2][1];
      if ( *(_BYTE *)(v21 + 84) )
        break;
      if ( sub_C8CA60(v21 + 56, *v18) )
        goto LABEL_30;
LABEL_45:
      if ( v19 == ++v18 )
        goto LABEL_18;
    }
    v22 = *(_QWORD **)(v21 + 64);
    v23 = &v22[*(unsigned int *)(v21 + 76)];
    if ( v22 == v23 )
      goto LABEL_45;
    while ( v20 != *v22 )
    {
      if ( v23 == ++v22 )
        goto LABEL_45;
    }
LABEL_30:
    v24 = *(_QWORD *)(v5 - 8);
    v25 = 0x1FFFFFFFE0LL;
    v26 = *(_DWORD *)(v5 + 72);
    v27 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
    if ( v27 )
    {
      v28 = 0;
      do
      {
        if ( v20 == *(_QWORD *)(v24 + 32LL * v26 + 8 * v28) )
        {
          v25 = 32 * v28;
          goto LABEL_35;
        }
        ++v28;
      }
      while ( v27 != (_DWORD)v28 );
      v29 = *(_QWORD *)(v24 + 0x1FFFFFFFE0LL);
      v30 = *a1[1];
      if ( v27 == v26 )
      {
LABEL_48:
        sub_B48D90(v5);
        v24 = *(_QWORD *)(v5 - 8);
        v27 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
      }
    }
    else
    {
LABEL_35:
      v29 = *(_QWORD *)(v24 + v25);
      v30 = *a1[1];
      if ( v27 == v26 )
        goto LABEL_48;
    }
    v31 = (v27 + 1) & 0x7FFFFFF;
    *(_DWORD *)(v5 + 4) = v31 | *(_DWORD *)(v5 + 4) & 0xF8000000;
    v32 = 32LL * (unsigned int)(v31 - 1) + v24;
    if ( *(_QWORD *)v32 )
    {
      v33 = *(_QWORD *)(v32 + 8);
      **(_QWORD **)(v32 + 16) = v33;
      if ( v33 )
        *(_QWORD *)(v33 + 16) = *(_QWORD *)(v32 + 16);
    }
    *(_QWORD *)v32 = v29;
    if ( v29 )
    {
      v34 = *(_QWORD *)(v29 + 16);
      *(_QWORD *)(v32 + 8) = v34;
      if ( v34 )
        *(_QWORD *)(v34 + 16) = v32 + 8;
      *(_QWORD *)(v32 + 16) = v29 + 16;
      *(_QWORD *)(v29 + 16) = v32;
    }
    *(_QWORD *)(*(_QWORD *)(v5 - 8) + 32LL * *(unsigned int *)(v5 + 72) + 8LL * ((*(_DWORD *)(v5 + 4) & 0x7FFFFFFu) - 1)) = v30;
LABEL_18:
    result = *(_QWORD *)(v5 + 32);
    if ( !result )
      BUG();
    v5 = 0;
    if ( *(_BYTE *)(result - 24) == 84 )
      v5 = result - 24;
  }
  while ( v35 != v5 );
  return result;
}
