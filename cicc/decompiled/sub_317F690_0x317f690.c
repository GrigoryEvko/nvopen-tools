// Function: sub_317F690
// Address: 0x317f690
//
__int64 __fastcall sub_317F690(_QWORD *a1, __int64 *a2, int *a3, size_t a4, char a5)
{
  unsigned __int64 *v5; // rax
  __int64 v10; // rbx
  _QWORD *v11; // r9
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  char *v14; // rdx
  _QWORD *v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rdx
  int v23; // ecx
  _QWORD *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  char v30; // [rsp+4h] [rbp-FCh]
  char *v31; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int64 *v32[2]; // [rsp+20h] [rbp-E0h] BYREF
  char **v33; // [rsp+30h] [rbp-D0h] BYREF
  int v34; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v35; // [rsp+40h] [rbp-C0h]
  int *v36; // [rsp+48h] [rbp-B8h]
  int *v37; // [rsp+50h] [rbp-B0h]
  __int64 v38; // [rsp+58h] [rbp-A8h]
  _QWORD *v39; // [rsp+60h] [rbp-A0h]
  __m128i v40; // [rsp+68h] [rbp-98h] BYREF
  __int64 v41; // [rsp+78h] [rbp-88h]
  __int64 v42; // [rsp+80h] [rbp-80h]
  __int64 v43; // [rsp+88h] [rbp-78h]

  v5 = (unsigned __int64 *)a4;
  v10 = *a2;
  if ( a3 )
  {
    v30 = a5;
    sub_C7D030(&v33);
    sub_C7D280((int *)&v33, a3, a4);
    sub_C7D290(&v33, v32);
    v5 = v32[0];
    a5 = v30;
  }
  v11 = (_QWORD *)a1[2];
  v12 = 33 * v10;
  v13 = a1 + 1;
  v14 = (char *)v5 + v12;
  v31 = v14;
  if ( !v11 )
    goto LABEL_10;
  v15 = a1 + 1;
  v16 = v11;
  do
  {
    while ( 1 )
    {
      v17 = v16[2];
      v18 = v16[3];
      if ( v16[4] >= (unsigned __int64)v14 )
        break;
      v16 = (_QWORD *)v16[3];
      if ( !v18 )
        goto LABEL_8;
    }
    v15 = v16;
    v16 = (_QWORD *)v16[2];
  }
  while ( v17 );
LABEL_8:
  if ( v13 != v15 && v15[4] <= (unsigned __int64)v14 )
    return (__int64)(v15 + 5);
LABEL_10:
  result = 0;
  if ( a5 )
  {
    v20 = *a2;
    BYTE4(v42) = 0;
    v40.m128i_i64[0] = (__int64)a3;
    v21 = (__int64)(a1 + 1);
    v34 = 0;
    v35 = 0;
    v36 = &v34;
    v37 = &v34;
    v38 = 0;
    v39 = a1;
    v40.m128i_i64[1] = a4;
    v41 = 0;
    v43 = v20;
    if ( !v11 )
      goto LABEL_12;
    do
    {
      while ( 1 )
      {
        v28 = v11[2];
        v29 = v11[3];
        if ( (unsigned __int64)v14 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v29 )
          goto LABEL_29;
      }
      v21 = (__int64)v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v28 );
LABEL_29:
    if ( v13 == (_QWORD *)v21 || (unsigned __int64)v14 < *(_QWORD *)(v21 + 32) )
    {
LABEL_12:
      v32[0] = (unsigned __int64 *)&v31;
      v21 = sub_317F580(a1, v21, v32);
    }
    sub_317D930(*(_QWORD **)(v21 + 56));
    *(_QWORD *)(v21 + 56) = 0;
    *(_QWORD *)(v21 + 64) = v21 + 48;
    *(_QWORD *)(v21 + 72) = v21 + 48;
    *(_QWORD *)(v21 + 80) = 0;
    v22 = v35;
    if ( v35 )
    {
      v23 = v34;
      *(_QWORD *)(v21 + 56) = v35;
      *(_DWORD *)(v21 + 48) = v23;
      *(_QWORD *)(v21 + 64) = v36;
      *(_QWORD *)(v21 + 72) = v37;
      *(_QWORD *)(v22 + 8) = v21 + 48;
      *(_QWORD *)(v21 + 80) = v38;
      v35 = 0;
      v36 = &v34;
      v37 = &v34;
      v38 = 0;
    }
    *(_QWORD *)(v21 + 88) = v39;
    *(__m128i *)(v21 + 96) = _mm_loadu_si128(&v40);
    *(_QWORD *)(v21 + 112) = v41;
    *(_QWORD *)(v21 + 120) = v42;
    *(_QWORD *)(v21 + 128) = v43;
    sub_317D930(0);
    v24 = (_QWORD *)a1[2];
    if ( v24 )
    {
      v25 = (__int64)(a1 + 1);
      do
      {
        while ( 1 )
        {
          v26 = v24[2];
          v27 = v24[3];
          if ( v24[4] >= (unsigned __int64)v31 )
            break;
          v24 = (_QWORD *)v24[3];
          if ( !v27 )
            goto LABEL_20;
        }
        v25 = (__int64)v24;
        v24 = (_QWORD *)v24[2];
      }
      while ( v26 );
LABEL_20:
      if ( v13 != (_QWORD *)v25 && (unsigned __int64)v31 >= *(_QWORD *)(v25 + 32) )
        return v25 + 40;
    }
    else
    {
      v25 = (__int64)(a1 + 1);
    }
    v33 = &v31;
    v25 = sub_317F580(a1, v25, (unsigned __int64 **)&v33);
    return v25 + 40;
  }
  return result;
}
