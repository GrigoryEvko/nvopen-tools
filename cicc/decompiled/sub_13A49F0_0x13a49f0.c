// Function: sub_13A49F0
// Address: 0x13a49f0
//
__int64 __fastcall sub_13A49F0(__int64 a1, unsigned int a2, unsigned __int8 a3, int a4, int a5, int a6)
{
  __int64 v8; // r15
  __int64 result; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r14
  __int64 v12; // r8
  int v13; // ecx
  unsigned int v14; // r9d
  int v15; // ecx
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned int v18; // r14d
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned int v23; // r12d
  int v24; // ecx
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r14
  unsigned int v28; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a1 + 8);
  if ( a2 <= (unsigned __int64)(v8 << 6) )
    goto LABEL_2;
  v10 = *(_QWORD *)a1;
  v11 = (a2 + 63) >> 6;
  if ( v11 < 2 * v8 )
    v11 = 2 * v8;
  v12 = (__int64)realloc(v10, 8 * v11, 8 * (int)v11, a4, a5, a6);
  if ( !v12 )
  {
    if ( 8 * v11 )
    {
      sub_16BD1C0("Allocation failed");
      v12 = 0;
    }
    else
    {
      v12 = sub_13A3880(1u);
    }
  }
  v13 = *(_DWORD *)(a1 + 16);
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v11;
  v14 = (unsigned int)(v13 + 63) >> 6;
  if ( v11 > v14 )
  {
    v27 = v11 - v14;
    if ( v27 )
    {
      v28 = (unsigned int)(v13 + 63) >> 6;
      memset((void *)(v12 + 8LL * v14), 0, 8 * v27);
      v13 = *(_DWORD *)(a1 + 16);
      v12 = *(_QWORD *)a1;
      v14 = v28;
    }
  }
  v15 = v13 & 0x3F;
  if ( v15 )
  {
    *(_QWORD *)(v12 + 8LL * (v14 - 1)) &= ~(-1LL << v15);
    v12 = *(_QWORD *)a1;
  }
  v16 = *(_QWORD *)(a1 + 8) - (unsigned int)v8;
  if ( v16 )
  {
    memset((void *)(v12 + 8LL * (unsigned int)v8), -a3, 8 * v16);
    result = *(unsigned int *)(a1 + 16);
    if ( (unsigned int)result >= a2 )
      goto LABEL_3;
  }
  else
  {
LABEL_2:
    result = *(unsigned int *)(a1 + 16);
    if ( (unsigned int)result >= a2 )
      goto LABEL_3;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = (unsigned int)(result + 63) >> 6;
  if ( v17 > v18 )
  {
    v25 = v17 - v18;
    if ( v25 )
    {
      memset((void *)(*(_QWORD *)a1 + 8LL * v18), -a3, 8 * v25);
      result = *(unsigned int *)(a1 + 16);
    }
  }
  if ( (result & 0x3F) == 0 )
  {
LABEL_3:
    *(_DWORD *)(a1 + 16) = a2;
    if ( a2 >= (unsigned int)result && !a3 )
      return result;
    goto LABEL_18;
  }
  v19 = -1LL << (result & 0x3F);
  v20 = (__int64 *)(*(_QWORD *)a1 + 8LL * (v18 - 1));
  v21 = *v20;
  if ( !a3 )
  {
    *v20 = v21 & ~v19;
    result = *(unsigned int *)(a1 + 16);
    goto LABEL_3;
  }
  *v20 = v21 | v19;
  *(_DWORD *)(a1 + 16) = a2;
LABEL_18:
  v22 = *(_QWORD *)(a1 + 8);
  v23 = (a2 + 63) >> 6;
  result = v23;
  if ( v22 > v23 )
  {
    v26 = v22 - v23;
    if ( v26 )
      result = (__int64)memset((void *)(*(_QWORD *)a1 + 8LL * v23), 0, 8 * v26);
  }
  v24 = *(_DWORD *)(a1 + 16) & 0x3F;
  if ( v24 )
  {
    result = ~(-1LL << v24);
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v23 - 1)) &= result;
  }
  return result;
}
