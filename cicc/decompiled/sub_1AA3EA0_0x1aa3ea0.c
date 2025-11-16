// Function: sub_1AA3EA0
// Address: 0x1aa3ea0
//
__int64 __fastcall sub_1AA3EA0(__int64 a1, unsigned int *a2, unsigned __int64 *a3, __int64 a4, int a5, int a6)
{
  void *v7; // rdi
  __int64 v8; // r15
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r12
  size_t v22; // rdx
  unsigned __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rdi
  const void *v27; // [rsp+8h] [rbp-38h]

  v7 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = 0x4000000000LL;
  v8 = *(_QWORD *)a2;
  v9 = *a3;
  v10 = *(_QWORD *)(*(_QWORD *)a2 + 40LL);
  v27 = v7;
  if ( v10 < *a3 )
  {
    v14 = v8 + 56LL * a2[2];
    if ( v14 != v8 )
    {
      v13 = 0;
      goto LABEL_21;
    }
    v13 = 0;
    v16 = a3[2] / v9;
    goto LABEL_34;
  }
  v11 = v10 / v9;
  v12 = 0;
  if ( v11 > 0x40 )
  {
    sub_16CD150(a1, v7, v11, 1, a5, a6);
    v12 = *(unsigned int *)(a1 + 8);
    v7 = (void *)(v12 + *(_QWORD *)a1);
  }
  if ( v11 != v12 )
    memset(v7, 241, v11 - v12);
  *(_DWORD *)(a1 + 8) = v11;
  v13 = (unsigned int)v11;
  v8 = *(_QWORD *)a2;
  v14 = *(_QWORD *)a2 + 56LL * a2[2];
  if ( v14 != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v8 + 40);
LABEL_21:
      v18 = (unsigned int)v13;
      v23 = v10 / v9;
      v24 = (unsigned int)v13;
      if ( v23 < (unsigned int)v13 )
      {
        *(_DWORD *)(a1 + 8) = v23;
        v13 = (unsigned int)v23;
        v18 = v23;
      }
      else if ( v23 > (unsigned int)v13 )
      {
        if ( v23 > *(unsigned int *)(a1 + 12) )
        {
          sub_16CD150(a1, v27, v23, 1, a5, a6);
          v24 = *(unsigned int *)(a1 + 8);
        }
        if ( v23 != v24 )
          memset((void *)(*(_QWORD *)a1 + v24), 242, v23 - v24);
        *(_DWORD *)(a1 + 8) = v23;
        v13 = (unsigned int)v23;
        v18 = (unsigned int)v23;
      }
      v19 = *(_QWORD *)(v8 + 8);
      v20 = v18 + v19 / v9;
      v21 = v20;
      if ( __CFADD__(v18, v19 / v9) )
        goto LABEL_17;
      if ( v20 > v18 )
        break;
LABEL_18:
      if ( v19 % v9 )
      {
        if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v13 )
        {
          sub_16CD150(a1, v27, 0, 1, a5, a6);
          v13 = *(unsigned int *)(a1 + 8);
        }
        *(_BYTE *)(*(_QWORD *)a1 + v13) = v19 % v9;
        v13 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
        *(_DWORD *)(a1 + 8) = v13;
      }
      v8 += 56;
      if ( v14 == v8 )
        goto LABEL_7;
    }
    if ( v20 > *(unsigned int *)(a1 + 12) )
    {
      sub_16CD150(a1, v27, v20, 1, a5, a6);
      v18 = *(unsigned int *)(a1 + 8);
      v22 = v21 - v18;
      if ( v21 == v18 )
        goto LABEL_17;
    }
    else
    {
      v22 = v20 - (unsigned int)v13;
      if ( v20 == (unsigned int)v13 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 8) = v21;
        v19 = *(_QWORD *)(v8 + 8);
        v13 = (unsigned int)v21;
        goto LABEL_18;
      }
    }
    memset((void *)(*(_QWORD *)a1 + v18), 0, v22);
    goto LABEL_17;
  }
LABEL_7:
  v15 = a3[2] / v9;
  v16 = v15;
  if ( v15 < v13 )
  {
    *(_DWORD *)(a1 + 8) = v15;
    return a1;
  }
LABEL_34:
  if ( v13 < v16 )
  {
    if ( v16 > *(unsigned int *)(a1 + 12) )
      sub_16CD150(a1, v27, v16, 1, a5, a6);
    v25 = *(unsigned int *)(a1 + 8);
    if ( v16 != v25 )
      memset((void *)(*(_QWORD *)a1 + v25), 243, v16 - v25);
    *(_DWORD *)(a1 + 8) = v16;
  }
  return a1;
}
