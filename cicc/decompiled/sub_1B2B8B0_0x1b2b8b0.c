// Function: sub_1B2B8B0
// Address: 0x1b2b8b0
//
__int64 __fastcall sub_1B2B8B0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // r15
  char v4; // bl
  __int64 v5; // r14
  __int64 *i; // rax
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // rdi
  __int64 *v10; // rcx
  unsigned __int64 v11; // r8
  __int64 *v12; // rdx
  __int64 *v13; // r13
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r13
  unsigned __int64 v20; // rdi
  unsigned int v21; // eax
  _QWORD *v22; // rbx
  _QWORD *v23; // r13
  __int64 v24; // r14
  __int64 result; // rax
  unsigned __int64 *j; // rbx
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // rcx
  __int64 *v29; // rax
  __int64 v30; // [rsp+8h] [rbp-168h]
  __int64 v31; // [rsp+10h] [rbp-160h] BYREF
  __int64 *v32; // [rsp+18h] [rbp-158h]
  __int64 *v33; // [rsp+20h] [rbp-150h]
  __int64 v34; // [rsp+28h] [rbp-148h]
  int v35; // [rsp+30h] [rbp-140h]
  _BYTE v36[312]; // [rsp+38h] [rbp-138h] BYREF

  v32 = (__int64 *)v36;
  v2 = *(_QWORD *)(a1 + 3480) == 0;
  v33 = (__int64 *)v36;
  v31 = 0;
  v34 = 32;
  v35 = 0;
  v30 = a1 + 3448;
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 3264);
    v4 = 1;
    v5 = v3 + 8LL * *(unsigned int *)(a1 + 3272);
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 3464);
    v5 = a1 + 3448;
    v4 = 0;
  }
  if ( v4 )
    goto LABEL_10;
LABEL_4:
  if ( v5 != v3 )
  {
    for ( i = (__int64 *)(v3 + 32); ; i = (__int64 *)v3 )
    {
      v7 = *i;
      v8 = v32;
      if ( v33 != v32 )
        break;
      v9 = &v32[HIDWORD(v34)];
      if ( v32 == v9 )
      {
LABEL_21:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
          break;
        ++HIDWORD(v34);
        *v9 = v7;
        ++v31;
      }
      else
      {
        v10 = 0;
        while ( v7 != *v8 )
        {
          if ( *v8 == -2 )
            v10 = v8;
          if ( v9 == ++v8 )
          {
            if ( !v10 )
              goto LABEL_21;
            *v10 = v7;
            --v35;
            ++v31;
            break;
          }
        }
      }
LABEL_8:
      if ( !v4 )
      {
        v3 = sub_220EF30(v3);
        goto LABEL_4;
      }
      v3 += 8;
LABEL_10:
      if ( v5 == v3 )
        goto LABEL_23;
    }
    sub_16CCBA0((__int64)&v31, v7);
    goto LABEL_8;
  }
LABEL_23:
  *(_DWORD *)(a1 + 3272) = 0;
  sub_1B2A3E0(*(_QWORD *)(a1 + 3456));
  v11 = (unsigned __int64)v33;
  *(_QWORD *)(a1 + 3456) = 0;
  v12 = v32;
  *(_QWORD *)(a1 + 3480) = 0;
  *(_QWORD *)(a1 + 3464) = v30;
  *(_QWORD *)(a1 + 3472) = v30;
  if ( (__int64 *)v11 == v12 )
    v13 = (__int64 *)(v11 + 8LL * HIDWORD(v34));
  else
    v13 = (__int64 *)(v11 + 8LL * (unsigned int)v34);
  if ( (__int64 *)v11 != v13 )
  {
    v14 = (__int64 *)v11;
    while ( 1 )
    {
      v15 = *v14;
      v16 = v14;
      if ( (unsigned __int64)*v14 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v13 == ++v14 )
        goto LABEL_29;
    }
    if ( v13 != v14 )
    {
      do
      {
        sub_15E3D00(v15);
        v29 = v16 + 1;
        if ( v16 + 1 == v13 )
          break;
        while ( 1 )
        {
          v15 = *v29;
          v16 = v29;
          if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v13 == ++v29 )
            goto LABEL_58;
        }
      }
      while ( v13 != v29 );
LABEL_58:
      v11 = (unsigned __int64)v33;
      v12 = v32;
    }
  }
LABEL_29:
  if ( v12 != (__int64 *)v11 )
    _libc_free(v11);
  sub_1B2A3E0(*(_QWORD *)(a1 + 3456));
  v17 = *(_QWORD *)(a1 + 3264);
  if ( v17 != a1 + 3280 )
    _libc_free(v17);
  j___libc_free_0(*(_QWORD *)(a1 + 3240));
  j___libc_free_0(*(_QWORD *)(a1 + 3208));
  v18 = *(unsigned __int64 **)(a1 + 112);
  v19 = &v18[12 * *(unsigned int *)(a1 + 120)];
  if ( v18 != v19 )
  {
    do
    {
      v19 -= 12;
      v20 = v19[6];
      if ( (unsigned __int64 *)v20 != v19 + 8 )
        _libc_free(v20);
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        _libc_free(*v19);
    }
    while ( v18 != v19 );
    v19 = *(unsigned __int64 **)(a1 + 112);
  }
  if ( v19 != (unsigned __int64 *)(a1 + 128) )
    _libc_free((unsigned __int64)v19);
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  v21 = *(_DWORD *)(a1 + 64);
  if ( v21 )
  {
    v22 = *(_QWORD **)(a1 + 48);
    v23 = &v22[2 * v21];
    do
    {
      if ( *v22 != -16 && *v22 != -8 )
      {
        v24 = v22[1];
        if ( v24 )
        {
          if ( (*(_BYTE *)(v24 + 8) & 1) == 0 )
            j___libc_free_0(*(_QWORD *)(v24 + 16));
          j_j___libc_free_0(v24, 552);
        }
      }
      v22 += 2;
    }
    while ( v23 != v22 );
  }
  result = j___libc_free_0(*(_QWORD *)(a1 + 48));
  for ( j = *(unsigned __int64 **)(a1 + 8);
        (unsigned __int64 *)a1 != j;
        result = (*(__int64 (__fastcall **)(unsigned __int64 *))(*(v27 - 1) + 8))(v27 - 1) )
  {
    v27 = j;
    j = (unsigned __int64 *)j[1];
    v28 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
    *j = v28 | *j & 7;
    *(_QWORD *)(v28 + 8) = j;
    v27[1] = 0;
    *v27 &= 7u;
  }
  return result;
}
