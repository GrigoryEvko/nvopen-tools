// Function: sub_1D1AA50
// Address: 0x1d1aa50
//
__int64 __fastcall sub_1D1AA50(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdx
  unsigned int v10; // r13d
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  int v14; // ecx
  unsigned int v15; // r8d
  int v16; // ecx
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned int v19; // eax
  unsigned __int64 v20; // rax
  unsigned int v21; // r14d
  int v22; // r13d
  int v23; // r9d
  __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // r8d
  unsigned int i; // ecx
  _DWORD *v28; // rax
  __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // r15d
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r14
  unsigned int v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    *(_DWORD *)(a2 + 16) = 0;
    v9 = *(unsigned int *)(a1 + 56);
    v10 = *(_DWORD *)(a1 + 56);
    if ( v9 > v8 << 6 )
    {
      v12 = (unsigned int)(v9 + 63) >> 6;
      if ( v12 < 2 * v8 )
        v12 = 2 * v8;
      v13 = (__int64)realloc(*(_QWORD *)a2, 8 * v12, 8 * (int)v12, a4, a5, a6);
      if ( !v13 && (8 * v12 || (v13 = malloc(1u)) == 0) )
      {
        v38 = v13;
        sub_16BD1C0("Allocation failed", 1u);
        v13 = v38;
      }
      v14 = *(_DWORD *)(a2 + 16);
      *(_QWORD *)a2 = v13;
      *(_QWORD *)(a2 + 8) = v12;
      v15 = (unsigned int)(v14 + 63) >> 6;
      if ( v15 < v12 )
      {
        v37 = (unsigned int)(v14 + 63) >> 6;
        memset((void *)(v13 + 8LL * v15), 0, 8 * (v12 - v15));
        v14 = *(_DWORD *)(a2 + 16);
        v13 = *(_QWORD *)a2;
        v15 = v37;
      }
      v16 = v14 & 0x3F;
      if ( v16 )
      {
        *(_QWORD *)(v13 + 8LL * (v15 - 1)) &= ~(-1LL << v16);
        v13 = *(_QWORD *)a2;
      }
      v17 = *(_QWORD *)(a2 + 8) - (unsigned int)v8;
      if ( v17 )
        memset((void *)(v13 + 8LL * (unsigned int)v8), 0, 8 * v17);
      v18 = *(_DWORD *)(a2 + 16);
      v19 = v18;
      if ( v10 <= v18 )
        goto LABEL_15;
      v36 = *(_QWORD *)(a2 + 8);
      v34 = (v18 + 63) >> 6;
      v33 = v34;
      if ( v34 >= v36 || (v8 = v36 - v34) == 0 )
      {
LABEL_36:
        v19 = v18;
        if ( (v18 & 0x3F) != 0 )
        {
          *(_QWORD *)(*(_QWORD *)a2 + 8LL * (v34 - 1)) &= ~(-1LL << (v18 & 0x3F));
          v19 = *(_DWORD *)(a2 + 16);
        }
LABEL_15:
        *(_DWORD *)(a2 + 16) = v10;
        if ( v19 > v10 )
        {
          v20 = *(_QWORD *)(a2 + 8);
          v21 = (v10 + 63) >> 6;
          if ( v20 > v21 )
          {
            v35 = v20 - v21;
            if ( v35 )
            {
              memset((void *)(*(_QWORD *)a2 + 8LL * v21), 0, 8 * v35);
              v10 = *(_DWORD *)(a2 + 16);
            }
          }
          v22 = v10 & 0x3F;
          if ( v22 )
            *(_QWORD *)(*(_QWORD *)a2 + 8LL * (v21 - 1)) &= ~(-1LL << v22);
        }
        goto LABEL_19;
      }
    }
    else
    {
      if ( !(_DWORD)v9 )
        return **(_QWORD **)(a1 + 32);
      v19 = 0;
      if ( !v8 )
        goto LABEL_15;
      v33 = 0;
      v34 = 0;
    }
    memset((void *)(*(_QWORD *)a2 + 8 * v33), 0, 8 * v8);
    v18 = *(_DWORD *)(a2 + 16);
    goto LABEL_36;
  }
LABEL_19:
  v23 = *(_DWORD *)(a1 + 56);
  if ( !v23 )
    return **(_QWORD **)(a1 + 32);
  v24 = 0;
  v25 = 0;
  v26 = 0;
  for ( i = 0; i != v23; ++i )
  {
    while ( 1 )
    {
      v28 = (_DWORD *)(v24 + *(_QWORD *)(a1 + 32));
      v29 = *(_QWORD *)v28;
      if ( *(_WORD *)(*(_QWORD *)v28 + 24LL) != 48 )
      {
        v30 = v28[2];
        if ( v25 )
        {
          if ( v30 != v26 || v29 != v25 )
            return 0;
        }
        else
        {
          v26 = v30;
          v25 = v29;
        }
        goto LABEL_27;
      }
      if ( a2 )
        break;
LABEL_27:
      ++i;
      v24 += 40;
      if ( i == v23 )
        goto LABEL_28;
    }
    v24 += 40;
    v31 = 1LL << i;
    v32 = i >> 6;
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v32) |= v31;
  }
LABEL_28:
  if ( !v25 )
    return **(_QWORD **)(a1 + 32);
  return v25;
}
