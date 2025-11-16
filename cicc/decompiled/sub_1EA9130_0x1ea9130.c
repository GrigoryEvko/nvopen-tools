// Function: sub_1EA9130
// Address: 0x1ea9130
//
unsigned __int64 __fastcall sub_1EA9130(_QWORD *a1, __int64 a2, int a3)
{
  unsigned __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rsi
  unsigned __int64 result; // rax
  __int64 *v19; // rax
  __int64 *v20; // rdi
  __int64 *v21; // rcx
  __int64 *v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // rdx
  __int64 *v25; // rsi
  __int64 *v26; // rdx
  unsigned __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+10h] [rbp-90h] BYREF
  __int64 *v29; // [rsp+18h] [rbp-88h]
  __int64 *v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+28h] [rbp-78h]
  int v32; // [rsp+30h] [rbp-70h]
  _BYTE v33[104]; // [rsp+38h] [rbp-68h] BYREF

  v4 = (unsigned __int64)(a1 + 3);
  if ( a1 + 3 == (_QWORD *)(a1[3] & 0xFFFFFFFFFFFFFFF8LL) )
    return a1[4];
  if ( !*(_BYTE *)(a2 + 180) )
    return sub_1DD5EE0((__int64)a1);
  v31 = 8;
  v29 = (__int64 *)v33;
  v30 = (__int64 *)v33;
  v5 = a1[7];
  v28 = 0;
  v32 = 0;
  v6 = *(_QWORD *)(v5 + 40);
  if ( a3 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 272) + 8LL * (unsigned int)a3);
  if ( !v7 )
  {
LABEL_58:
    v17 = a1[4];
    goto LABEL_25;
  }
  v8 = *(_QWORD *)(v7 + 16);
  if ( a1 != *(_QWORD **)(v8 + 24) )
    goto LABEL_9;
LABEL_33:
  v19 = v29;
  if ( v30 != v29 )
  {
LABEL_34:
    sub_16CCBA0((__int64)&v28, v8);
    goto LABEL_35;
  }
  v20 = &v29[HIDWORD(v31)];
  if ( v29 == v20 )
  {
LABEL_71:
    if ( HIDWORD(v31) < (unsigned int)v31 )
    {
      ++HIDWORD(v31);
      *v20 = v8;
      ++v28;
      goto LABEL_35;
    }
    goto LABEL_34;
  }
  v21 = 0;
  while ( *v19 != v8 )
  {
    if ( *v19 == -2 )
      v21 = v19;
    if ( v20 == ++v19 )
    {
      if ( !v21 )
        goto LABEL_71;
      *v21 = v8;
      --v32;
      ++v28;
      break;
    }
  }
LABEL_35:
  v8 = *(_QWORD *)(v7 + 16);
LABEL_9:
  while ( 1 )
  {
    v7 = *(_QWORD *)(v7 + 32);
    if ( !v7 )
      break;
    v9 = *(_QWORD *)(v7 + 16);
    if ( v9 != v8 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      if ( a1 == *(_QWORD **)(v9 + 24) )
        goto LABEL_33;
    }
  }
  v10 = HIDWORD(v31);
  if ( HIDWORD(v31) == v32 )
    goto LABEL_58;
  if ( HIDWORD(v31) - v32 != 1 )
  {
    while ( 1 )
    {
      v11 = (_QWORD *)(*(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v11;
      if ( !v11 )
        goto LABEL_74;
      v4 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = *v11;
      if ( (v13 & 4) == 0 && (*((_BYTE *)v12 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
          v4 = v14;
          if ( (*(_BYTE *)(v14 + 46) & 4) == 0 )
            break;
          v13 = *(_QWORD *)v14;
        }
      }
      v15 = v29;
      if ( v30 == v29 )
      {
        v16 = &v29[HIDWORD(v31)];
        if ( v29 == v16 )
        {
          v26 = v29;
        }
        else
        {
          do
          {
            if ( v4 == *v15 )
              break;
            ++v15;
          }
          while ( v16 != v15 );
          v26 = &v29[HIDWORD(v31)];
        }
        goto LABEL_43;
      }
      v16 = &v30[(unsigned int)v31];
      v15 = sub_16CC9F0((__int64)&v28, v4);
      if ( v4 == *v15 )
        break;
      if ( v30 == v29 )
      {
        v15 = &v30[HIDWORD(v31)];
        v26 = v15;
        goto LABEL_43;
      }
      v15 = &v30[(unsigned int)v31];
LABEL_22:
      if ( v16 != v15 )
      {
        if ( (*(_BYTE *)v4 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
            v4 = *(_QWORD *)(v4 + 8);
        }
        v17 = *(_QWORD *)(v4 + 8);
        goto LABEL_25;
      }
    }
    if ( v30 == v29 )
      v26 = &v30[HIDWORD(v31)];
    else
      v26 = &v30[(unsigned int)v31];
LABEL_43:
    while ( v26 != v15 && (unsigned __int64)*v15 >= 0xFFFFFFFFFFFFFFFELL )
      ++v15;
    goto LABEL_22;
  }
  v22 = v30;
  if ( v30 != v29 )
    v10 = (unsigned int)v31;
  v23 = &v30[v10];
  v24 = *v30;
  if ( v30 != v23 )
  {
    while ( 1 )
    {
      v24 = *v22;
      v25 = v22;
      if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v23 == ++v22 )
      {
        v24 = v25[1];
        break;
      }
    }
  }
  if ( !v24 )
LABEL_74:
    BUG();
  if ( (*(_BYTE *)v24 & 4) == 0 )
  {
    while ( (*(_BYTE *)(v24 + 46) & 8) != 0 )
      v24 = *(_QWORD *)(v24 + 8);
  }
  v17 = *(_QWORD *)(v24 + 8);
LABEL_25:
  result = sub_1DD5D40((__int64)a1, v17);
  if ( v30 != v29 )
  {
    v27 = result;
    _libc_free((unsigned __int64)v30);
    return v27;
  }
  return result;
}
