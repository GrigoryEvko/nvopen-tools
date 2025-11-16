// Function: sub_2ECD150
// Address: 0x2ecd150
//
unsigned __int64 __fastcall sub_2ECD150(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 *i; // r12
  __int64 *v7; // rsi
  char *v8; // r14
  char *v9; // r13
  size_t v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // r14
  __int64 v13; // r12
  unsigned __int64 result; // rax
  __int64 v15; // r13
  __int64 v16; // r15
  char *v17; // rsi
  _BYTE *v18; // rdi
  char *v19; // rax
  char *v20; // rcx
  unsigned __int64 v21; // rdi
  char *v22; // rcx
  char *v23; // rcx
  size_t v24; // rdx
  _BYTE *v25; // rax
  char *v26; // r15
  char *v27; // r15
  char *v28; // [rsp+18h] [rbp-108h]
  _BYTE *v29; // [rsp+20h] [rbp-100h] BYREF
  __int64 v30; // [rsp+28h] [rbp-F8h]
  _BYTE v31[240]; // [rsp+30h] [rbp-F0h] BYREF

  *(_QWORD *)(a1 + 3992) = 0xFFFFFFFFLL;
  v2 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 3648) = 0;
  v3 = *(_DWORD *)(v2 + 64);
  if ( v3 < *(_DWORD *)(a1 + 3984) >> 2 || v3 > *(_DWORD *)(a1 + 3984) )
  {
    _libc_free(*(_QWORD *)(a1 + 3976));
    v4 = (__int64)_libc_calloc(v3, 1u);
    if ( !v4 && (v3 || (v4 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    *(_QWORD *)(a1 + 3976) = v4;
    *(_DWORD *)(a1 + 3984) = v3;
  }
  v5 = *(__int64 **)(a1 + 56);
  for ( i = *(__int64 **)(a1 + 48); v5 != i; i += 32 )
  {
    v7 = i;
    sub_2ECAAF0(a1, v7);
  }
  sub_2F796A0(
    a1 + 5376,
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 3544),
    *(_QWORD *)(a1 + 3464),
    *(_QWORD *)(a1 + 904),
    *(_QWORD *)(a1 + 912),
    *(unsigned __int8 *)(a1 + 4017),
    0);
  sub_2F796A0(
    a1 + 6248,
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 3544),
    *(_QWORD *)(a1 + 3464),
    *(_QWORD *)(a1 + 904),
    *(_QWORD *)(a1 + 3632),
    *(unsigned __int8 *)(a1 + 4017),
    0);
  sub_2F75920(a1 + 4480);
  sub_2F76D20(a1 + 5376, *(_QWORD *)(*(_QWORD *)(a1 + 4528) + 24LL), *(unsigned int *)(*(_QWORD *)(a1 + 4528) + 32LL));
  sub_2F76D20(a1 + 6248, *(_QWORD *)(*(_QWORD *)(a1 + 4528) + 232LL), *(unsigned int *)(*(_QWORD *)(a1 + 4528) + 240LL));
  sub_2F75570(a1 + 5376);
  sub_2F75730(a1 + 6248);
  sub_2F79800(a1 + 6248, a1 + 4480);
  v8 = *(char **)(a1 + 6648);
  v9 = *(char **)(a1 + 6640);
  v10 = v8 - v9;
  if ( v8 != v9 )
  {
    v18 = *(_BYTE **)(a1 + 5768);
    if ( v10 <= *(_QWORD *)(a1 + 5784) - (_QWORD)v18 )
    {
      v23 = *(char **)(a1 + 5776);
      v24 = v23 - v18;
      if ( v10 > v23 - v18 )
      {
        v27 = &v9[v24];
        if ( v9 != &v9[v24] )
        {
          memmove(v18, *(const void **)(a1 + 6640), v24);
          v23 = *(char **)(a1 + 5776);
        }
        if ( v8 != v27 )
          v23 = (char *)memmove(v23, v27, v8 - v27);
        *(_QWORD *)(a1 + 5776) = &v23[v8 - v27];
      }
      else
      {
        if ( v9 != v8 )
        {
          v25 = memmove(v18, *(const void **)(a1 + 6640), v10);
          v23 = *(char **)(a1 + 5776);
          v18 = v25;
        }
        v26 = &v18[v10];
        if ( v23 != v26 )
          *(_QWORD *)(a1 + 5776) = v26;
      }
    }
    else
    {
      if ( v10 > 0x7FFFFFFFFFFFFFFCLL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v19 = (char *)sub_22077B0(v10);
      v20 = v19;
      if ( v9 != v8 )
        v20 = (char *)memcpy(v19, v9, v10);
      v21 = *(_QWORD *)(a1 + 5768);
      if ( v21 )
      {
        v28 = v20;
        j_j___libc_free_0(v21);
        v20 = v28;
      }
      *(_QWORD *)(a1 + 5768) = v20;
      v22 = &v20[v10];
      *(_QWORD *)(a1 + 5776) = v22;
      *(_QWORD *)(a1 + 5784) = v22;
    }
  }
  sub_2EC6D00(a1, *(_QWORD *)(*(_QWORD *)(a1 + 4528) + 232LL), *(unsigned int *)(*(_QWORD *)(a1 + 4528) + 240LL));
  if ( *(_QWORD *)(a1 + 3632) != *(_QWORD *)(a1 + 920) )
  {
    v30 = 0x800000000LL;
    v29 = v31;
    sub_2F78B80(a1 + 6248, &v29);
    sub_2EC6D00(a1, (__int64)v29, (unsigned int)v30);
    if ( v29 != v31 )
      _libc_free((unsigned __int64)v29);
  }
  v11 = *(_QWORD *)(a1 + 4896);
  if ( v11 != *(_QWORD *)(a1 + 4904) )
    *(_QWORD *)(a1 + 4904) = v11;
  v12 = *(_QWORD **)(a1 + 4528);
  v13 = 0;
  result = (__int64)(v12[1] - *v12) >> 2;
  v15 = (unsigned int)result;
  if ( (_DWORD)result )
  {
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(a1 + 3544);
        result = *(unsigned int *)(*(_QWORD *)(v16 + 296) + 4 * v13);
        if ( !(_DWORD)result )
          break;
        if ( *(_DWORD *)(*v12 + 4 * v13) > (unsigned int)result )
          goto LABEL_19;
LABEL_16:
        if ( ++v13 == v15 )
          return result;
      }
      *(_DWORD *)(*(_QWORD *)(v16 + 296) + 4 * v13) = sub_2F60A40(*(_QWORD *)(a1 + 3544));
      result = *(unsigned int *)(*(_QWORD *)(v16 + 296) + 4 * v13);
      if ( *(_DWORD *)(*v12 + 4 * v13) <= (unsigned int)result )
        goto LABEL_16;
LABEL_19:
      v17 = *(char **)(a1 + 4904);
      LODWORD(v29) = (unsigned __int16)(v13 + 1);
      result = 0;
      if ( v17 == *(char **)(a1 + 4912) )
      {
        result = sub_2ECCFD0((unsigned __int64 *)(a1 + 4896), v17, &v29);
        goto LABEL_16;
      }
      if ( v17 )
      {
        result = (unsigned int)v29;
        *(_DWORD *)v17 = (_DWORD)v29;
        v17 = *(char **)(a1 + 4904);
      }
      ++v13;
      *(_QWORD *)(a1 + 4904) = v17 + 4;
    }
    while ( v13 != v15 );
  }
  return result;
}
