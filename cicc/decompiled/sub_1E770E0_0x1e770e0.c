// Function: sub_1E770E0
// Address: 0x1e770e0
//
__int64 __fastcall sub_1E770E0(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 i; // r12
  __int64 v7; // rsi
  char *v8; // rcx
  __int64 *v9; // r8
  char *v10; // r14
  char *v11; // r13
  size_t v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 *v15; // r8
  __int64 v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // r12
  __int64 result; // rax
  __int64 v20; // r13
  __int64 v21; // rdx
  char *v22; // rsi
  _BYTE *v23; // rdi
  char *v24; // rax
  char *v25; // rcx
  __int64 v26; // rdi
  size_t v27; // rdx
  _BYTE *v28; // rax
  char *v29; // r15
  char *v30; // r15
  _DWORD *v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  char *v33; // [rsp+18h] [rbp-88h]
  unsigned int *v34; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h]
  _BYTE v36[112]; // [rsp+30h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 2544) = 0xFFFFFFFFLL;
  v2 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 2328) = 0;
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 < *(_DWORD *)(a1 + 2536) >> 2 || v3 > *(_DWORD *)(a1 + 2536) )
  {
    _libc_free(*(_QWORD *)(a1 + 2528));
    v4 = (__int64)_libc_calloc(v3, 1u);
    if ( !v4 )
    {
      if ( v3 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v4 = 0;
      }
      else
      {
        v4 = sub_13A3880(1u);
      }
    }
    *(_QWORD *)(a1 + 2528) = v4;
    *(_DWORD *)(a1 + 2536) = v3;
  }
  v5 = *(_QWORD *)(a1 + 56);
  for ( i = *(_QWORD *)(a1 + 48); v5 != i; i += 272 )
  {
    v7 = i;
    sub_1E75140(a1, v7);
  }
  sub_1EE96B0(
    a1 + 3288,
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 2272),
    *(_QWORD *)(a1 + 2112),
    *(_QWORD *)(a1 + 920),
    *(_QWORD *)(a1 + 928),
    *(unsigned __int8 *)(a1 + 2569),
    0);
  sub_1EE96B0(
    a1 + 3776,
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 2272),
    *(_QWORD *)(a1 + 2112),
    *(_QWORD *)(a1 + 920),
    *(_QWORD *)(a1 + 2312),
    *(unsigned __int8 *)(a1 + 2569),
    0);
  sub_1EE6590(a1 + 2776);
  sub_1EE72E0(a1 + 3288, *(_QWORD *)(*(_QWORD *)(a1 + 2824) + 24LL), *(unsigned int *)(*(_QWORD *)(a1 + 2824) + 32LL));
  sub_1EE72E0(a1 + 3776, *(_QWORD *)(*(_QWORD *)(a1 + 2824) + 104LL), *(unsigned int *)(*(_QWORD *)(a1 + 2824) + 112LL));
  sub_1EE6350(a1 + 3288);
  sub_1EE6470(a1 + 3776);
  sub_1EE9830(a1 + 3776, a1 + 2776);
  v10 = *(char **)(a1 + 4048);
  v11 = *(char **)(a1 + 4040);
  v12 = v10 - v11;
  if ( v10 != v11 )
  {
    v23 = *(_BYTE **)(a1 + 3552);
    if ( v12 <= *(_QWORD *)(a1 + 3568) - (_QWORD)v23 )
    {
      v8 = *(char **)(a1 + 3560);
      v27 = v8 - v23;
      if ( v12 > v8 - v23 )
      {
        v30 = &v11[v27];
        if ( v11 != &v11[v27] )
        {
          memmove(v23, *(const void **)(a1 + 4040), v27);
          v8 = *(char **)(a1 + 3560);
        }
        if ( v10 != v30 )
          v8 = (char *)memmove(v8, v30, v10 - v30);
        v8 += v10 - v30;
        *(_QWORD *)(a1 + 3560) = v8;
      }
      else
      {
        if ( v11 != v10 )
        {
          v28 = memmove(v23, *(const void **)(a1 + 4040), v12);
          v8 = *(char **)(a1 + 3560);
          v23 = v28;
        }
        v29 = &v23[v12];
        if ( v8 != v29 )
          *(_QWORD *)(a1 + 3560) = v29;
      }
    }
    else
    {
      if ( v12 > 0x7FFFFFFFFFFFFFFCLL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v24 = (char *)sub_22077B0(v12);
      v25 = v24;
      if ( v11 != v10 )
        v25 = (char *)memcpy(v24, v11, v12);
      v26 = *(_QWORD *)(a1 + 3552);
      if ( v26 )
      {
        v33 = v25;
        j_j___libc_free_0(v26, *(_QWORD *)(a1 + 3568) - v26);
        v25 = v33;
      }
      *(_QWORD *)(a1 + 3552) = v25;
      v8 = &v25[v12];
      *(_QWORD *)(a1 + 3560) = v8;
      *(_QWORD *)(a1 + 3568) = v8;
    }
  }
  sub_1E70F80(
    (_QWORD *)a1,
    *(unsigned int **)(*(_QWORD *)(a1 + 2824) + 104LL),
    *(unsigned int *)(*(_QWORD *)(a1 + 2824) + 112LL),
    (__int64)v8,
    v9);
  if ( *(_QWORD *)(a1 + 2312) != *(_QWORD *)(a1 + 936) )
  {
    v35 = 0x800000000LL;
    v34 = (unsigned int *)v36;
    sub_1EE8D00(a1 + 3776, &v34, v13);
    sub_1E70F80((_QWORD *)a1, v34, (unsigned int)v35, v14, v15);
    if ( v34 != (unsigned int *)v36 )
      _libc_free((unsigned __int64)v34);
  }
  v16 = *(_QWORD *)(a1 + 3064);
  if ( v16 != *(_QWORD *)(a1 + 3072) )
    *(_QWORD *)(a1 + 3072) = v16;
  v17 = *(_QWORD **)(a1 + 2824);
  v18 = 0;
  result = (__int64)(v17[1] - *v17) >> 2;
  v20 = (unsigned int)result;
  if ( (_DWORD)result )
  {
    do
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(a1 + 2272);
        result = *(unsigned int *)(4 * v18 + *(_QWORD *)(v21 + 88));
        if ( !(_DWORD)result )
          break;
        if ( *(_DWORD *)(*v17 + 4 * v18) > (unsigned int)result )
          goto LABEL_19;
LABEL_16:
        if ( ++v18 == v20 )
          return result;
      }
      v31 = (_DWORD *)(4 * v18 + *(_QWORD *)(v21 + 88));
      v32 = *(_QWORD *)(a1 + 2272);
      *v31 = sub_1ED7BB0(v32, (unsigned int)v18);
      result = *(unsigned int *)(*(_QWORD *)(v32 + 88) + 4 * v18);
      if ( *(_DWORD *)(*v17 + 4 * v18) <= (unsigned int)result )
        goto LABEL_16;
LABEL_19:
      v22 = *(char **)(a1 + 3072);
      LODWORD(v34) = (unsigned __int16)(v18 + 1);
      result = 0;
      if ( v22 == *(char **)(a1 + 3080) )
      {
        result = sub_1E76F60((char **)(a1 + 3064), v22, &v34);
        goto LABEL_16;
      }
      if ( v22 )
      {
        result = (unsigned int)v34;
        *(_DWORD *)v22 = (_DWORD)v34;
        v22 = *(char **)(a1 + 3072);
      }
      ++v18;
      *(_QWORD *)(a1 + 3072) = v22 + 4;
    }
    while ( v18 != v20 );
  }
  return result;
}
