// Function: sub_1F130D0
// Address: 0x1f130d0
//
__int64 __fastcall sub_1F130D0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned __int64 v8; // rdx
  unsigned int v9; // r12d
  __int64 result; // rax
  unsigned __int64 v11; // r13
  __int64 v12; // r15
  int v13; // ecx
  unsigned int v14; // r8d
  int v15; // ecx
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned int v18; // r13d
  unsigned __int64 v19; // rdx
  unsigned int v20; // r13d
  int v21; // r12d
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r13
  unsigned int v25; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 272) = a2;
  *(_DWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  *(_DWORD *)(a2 + 16) = 0;
  v6 = *(_QWORD *)(a1 + 272);
  v7 = *(_QWORD *)(v6 + 8);
  v8 = *(unsigned int *)(*(_QWORD *)(a1 + 240) + 288LL);
  v9 = *(_DWORD *)(*(_QWORD *)(a1 + 240) + 288LL);
  if ( v8 <= v7 << 6 )
    goto LABEL_2;
  v11 = (unsigned int)(v8 + 63) >> 6;
  if ( v11 < 2 * v7 )
    v11 = 2 * v7;
  v12 = (__int64)realloc(*(_QWORD *)v6, 8 * v11, 8 * (int)v11, a4, a5, a6);
  if ( !v12 )
  {
    if ( 8 * v11 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      v12 = sub_13A3880(1u);
  }
  v13 = *(_DWORD *)(v6 + 16);
  *(_QWORD *)v6 = v12;
  *(_QWORD *)(v6 + 8) = v11;
  v14 = (unsigned int)(v13 + 63) >> 6;
  if ( v11 > v14 )
  {
    v24 = v11 - v14;
    if ( v24 )
    {
      v25 = (unsigned int)(v13 + 63) >> 6;
      memset((void *)(v12 + 8LL * v14), 0, 8 * v24);
      v13 = *(_DWORD *)(v6 + 16);
      v12 = *(_QWORD *)v6;
      v14 = v25;
    }
  }
  v15 = v13 & 0x3F;
  if ( v15 )
  {
    *(_QWORD *)(v12 + 8LL * (v14 - 1)) &= ~(-1LL << v15);
    v12 = *(_QWORD *)v6;
  }
  v16 = *(_QWORD *)(v6 + 8) - (unsigned int)v7;
  if ( v16 )
  {
    memset((void *)(v12 + 8LL * (unsigned int)v7), 0, 8 * v16);
    result = *(unsigned int *)(v6 + 16);
    if ( v9 <= (unsigned int)result )
      goto LABEL_3;
  }
  else
  {
LABEL_2:
    result = *(unsigned int *)(v6 + 16);
    if ( v9 <= (unsigned int)result )
      goto LABEL_3;
  }
  v17 = *(_QWORD *)(v6 + 8);
  v18 = (unsigned int)(result + 63) >> 6;
  if ( v17 > v18 )
  {
    v22 = v17 - v18;
    if ( v22 )
    {
      memset((void *)(*(_QWORD *)v6 + 8LL * v18), 0, 8 * v22);
      result = *(unsigned int *)(v6 + 16);
    }
  }
  if ( (result & 0x3F) != 0 )
  {
    *(_QWORD *)(*(_QWORD *)v6 + 8LL * (v18 - 1)) &= ~(-1LL << (result & 0x3F));
    result = *(unsigned int *)(v6 + 16);
    *(_DWORD *)(v6 + 16) = v9;
    if ( v9 >= (unsigned int)result )
      return result;
    goto LABEL_16;
  }
LABEL_3:
  *(_DWORD *)(v6 + 16) = v9;
  if ( v9 >= (unsigned int)result )
    return result;
LABEL_16:
  v19 = *(_QWORD *)(v6 + 8);
  v20 = (v9 + 63) >> 6;
  result = v20;
  if ( v19 > v20 )
  {
    v23 = v19 - v20;
    if ( v23 )
    {
      result = (__int64)memset((void *)(*(_QWORD *)v6 + 8LL * v20), 0, 8 * v23);
      v9 = *(_DWORD *)(v6 + 16);
    }
  }
  v21 = v9 & 0x3F;
  if ( v21 )
  {
    result = ~(-1LL << v21);
    *(_QWORD *)(*(_QWORD *)v6 + 8LL * (v20 - 1)) &= result;
  }
  return result;
}
