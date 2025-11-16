// Function: sub_1E75FC0
// Address: 0x1e75fc0
//
unsigned __int64 __fastcall sub_1E75FC0(__int64 a1)
{
  _QWORD *v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r15
  unsigned __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // r9
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r14
  __int64 v19; // rbx
  int v20; // r15d
  unsigned __int64 result; // rax
  __int64 v22; // rax
  unsigned int v23; // ebx
  int v24; // r15d
  unsigned __int64 v25; // rdx
  unsigned int v26; // r13d
  unsigned __int64 v27; // r13
  __int64 v28; // rax
  int v29; // ecx
  unsigned int v30; // r8d
  int v31; // ecx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  unsigned __int64 v35; // rdx
  unsigned int v36; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 2280);
  if ( v2 )
  {
    v3 = v2[1];
    v4 = v2[2];
    v5 = (unsigned __int64 *)v2[22];
    v6 = (unsigned __int64 *)v2[23];
  }
  else
  {
    v33 = (_BYTE *)sub_22077B0(224);
    v2 = v33;
    if ( v33 )
    {
      *v33 = 1;
      v34 = v33 + 48;
      v6 = 0;
      v5 = 0;
      *((_DWORD *)v34 - 11) = 8;
      v4 = 0;
      *((_QWORD *)v34 - 5) = 0;
      *((_QWORD *)v34 - 4) = 0;
      *((_QWORD *)v34 - 3) = 0;
      v2[22] = 0;
      v2[23] = 0;
      v2[24] = 0;
      v2[25] = 0;
      v2[26] = 0;
      v2[27] = 0;
      v2[4] = v34;
      v2[5] = 0x1000000000LL;
      v3 = 0;
    }
    else
    {
      v3 = MEMORY[8];
      v4 = MEMORY[0x10];
      v5 = (unsigned __int64 *)MEMORY[0xB0];
      v6 = (unsigned __int64 *)MEMORY[0xB8];
    }
    *(_QWORD *)(a1 + 2280) = v2;
  }
  if ( v4 != v3 )
    v2[2] = v3;
  *((_DWORD *)v2 + 10) = 0;
  if ( v5 != v6 )
  {
    v7 = v5;
    do
    {
      if ( (unsigned __int64 *)*v7 != v7 + 2 )
        _libc_free(*v7);
      v7 += 6;
    }
    while ( v7 != v6 );
    v2[23] = v5;
  }
  v8 = v2[25];
  if ( v8 != v2[26] )
    v2[26] = v8;
  v9 = *(_QWORD *)(a1 + 48);
  v10 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 2304) = 0;
  v11 = *(_QWORD *)(a1 + 2280);
  v12 = *(_QWORD *)(v11 + 8);
  v13 = 0xF0F0F0F0F0F0F0F1LL * ((v10 - v9) >> 4);
  v14 = (*(_QWORD *)(v11 + 16) - v12) >> 3;
  if ( (unsigned int)v13 > v14 )
  {
    sub_1E75E10((char **)(v11 + 8), (unsigned int)v13 - v14);
  }
  else if ( (unsigned int)v13 < v14 )
  {
    v22 = v12 + 8LL * (unsigned int)v13;
    if ( *(_QWORD *)(v11 + 16) != v22 )
      *(_QWORD *)(v11 + 16) = v22;
  }
  sub_1F07D80();
  v18 = *(_QWORD *)(a1 + 2296);
  v19 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 2280) + 208LL) - *(_QWORD *)(*(_QWORD *)(a1 + 2280) + 200LL)) >> 2;
  LOBYTE(v20) = v19;
  if ( (unsigned int)v19 > (unsigned __int64)(v18 << 6) )
  {
    v27 = (unsigned int)(v19 + 63) >> 6;
    if ( v27 < 2 * v18 )
      v27 = 2 * v18;
    v28 = (__int64)realloc(*(_QWORD *)(a1 + 2288), 8 * v27, 8 * (int)v27, v15, v16, v17);
    if ( !v28 )
    {
      if ( 8 * v27 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v28 = 0;
      }
      else
      {
        v28 = sub_13A3880(1u);
      }
    }
    v29 = *(_DWORD *)(a1 + 2304);
    *(_QWORD *)(a1 + 2288) = v28;
    *(_QWORD *)(a1 + 2296) = v27;
    v30 = (unsigned int)(v29 + 63) >> 6;
    if ( v27 > v30 )
    {
      v36 = (unsigned int)(v29 + 63) >> 6;
      memset((void *)(v28 + 8LL * v30), 0, 8 * (v27 - v30));
      v29 = *(_DWORD *)(a1 + 2304);
      v30 = v36;
      v28 = *(_QWORD *)(a1 + 2288);
    }
    v31 = v29 & 0x3F;
    if ( v31 )
    {
      *(_QWORD *)(v28 + 8LL * (v30 - 1)) &= ~(-1LL << v31);
      v28 = *(_QWORD *)(a1 + 2288);
    }
    v32 = *(_QWORD *)(a1 + 2296) - (unsigned int)v18;
    if ( v32 )
      memset((void *)(v28 + 8LL * (unsigned int)v18), 0, 8 * v32);
  }
  result = *(unsigned int *)(a1 + 2304);
  if ( (unsigned int)v19 > (unsigned int)result )
  {
    v25 = *(_QWORD *)(a1 + 2296);
    v26 = (unsigned int)(result + 63) >> 6;
    if ( v25 > v26 )
    {
      v35 = v25 - v26;
      if ( v35 )
      {
        memset((void *)(*(_QWORD *)(a1 + 2288) + 8LL * v26), 0, 8 * v35);
        result = *(unsigned int *)(a1 + 2304);
      }
    }
    if ( (result & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 2288) + 8LL * (v26 - 1)) &= ~(-1LL << (result & 0x3F));
      result = *(unsigned int *)(a1 + 2304);
    }
  }
  *(_DWORD *)(a1 + 2304) = v19;
  if ( (unsigned int)v19 < (unsigned int)result )
  {
    result = *(_QWORD *)(a1 + 2296);
    v23 = (unsigned int)(v19 + 63) >> 6;
    if ( result > v23 )
    {
      result -= v23;
      if ( result )
      {
        result = (unsigned __int64)memset((void *)(*(_QWORD *)(a1 + 2288) + 8LL * v23), 0, 8 * result);
        v20 = *(_DWORD *)(a1 + 2304);
      }
    }
    v24 = v20 & 0x3F;
    if ( v24 )
    {
      result = ~(-1LL << v24);
      *(_QWORD *)(*(_QWORD *)(a1 + 2288) + 8LL * (v23 - 1)) &= result;
    }
  }
  return result;
}
