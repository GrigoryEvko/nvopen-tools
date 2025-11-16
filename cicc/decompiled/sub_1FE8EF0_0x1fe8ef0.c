// Function: sub_1FE8EF0
// Address: 0x1fe8ef0
//
unsigned __int64 *__fastcall sub_1FE8EF0(size_t *a1, unsigned __int64 a2, __int64 a3)
{
  __int32 v5; // r9d
  __int64 v6; // rax
  bool v7; // cc
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r8
  int v11; // r9d
  __int32 v12; // eax
  size_t v13; // r10
  __int64 v14; // r14
  __int64 *v15; // r15
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // r8
  int v21; // r9d
  unsigned __int64 *result; // rax
  unsigned int i; // ecx
  unsigned __int64 **v24; // rdx
  unsigned __int64 *v25; // r15
  unsigned int v26; // ecx
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  int v30; // r9d
  unsigned __int64 *v31; // r8
  unsigned int v32; // edx
  unsigned int v33; // edx
  int v34; // r10d
  int v35; // ecx
  int v36; // ecx
  int v37; // eax
  int v38; // edx
  __int64 v39; // rsi
  int v40; // edi
  unsigned int j; // r12d
  unsigned int v42; // r12d
  int v43; // esi
  int v44; // ecx
  size_t v45; // [rsp+0h] [rbp-70h]
  __int32 v46; // [rsp+8h] [rbp-68h]
  __int32 v47; // [rsp+Ch] [rbp-64h]
  __m128i v48; // [rsp+10h] [rbp-60h] BYREF
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+28h] [rbp-48h]
  __int64 v51; // [rsp+30h] [rbp-40h]

  v5 = sub_1FE6610(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3);
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v7 = *(_DWORD *)(v6 + 32) <= 0x40u;
  v8 = *(_QWORD **)(v6 + 24);
  if ( !v7 )
    v8 = (_QWORD *)*v8;
  v46 = v5;
  v9 = sub_1F4AAF0(a1[3], *(_QWORD **)(*(_QWORD *)(a1[3] + 256) + 8LL * (unsigned int)v8));
  v12 = sub_1E6B9A0(a1[1], (__int64)v9, (unsigned __int8 *)byte_3F871B3, 0, v10, v11);
  v13 = a1[5];
  v47 = v12;
  v14 = *(_QWORD *)(v13 + 56);
  v15 = (__int64 *)a1[6];
  v45 = v13;
  v16 = (__int64)sub_1E0B640(v14, *(_QWORD *)(a1[2] + 8) + 960LL, (__int64 *)(a2 + 72), 0);
  sub_1DD5BA0((__int64 *)(v45 + 16), v16);
  v17 = *v15;
  v18 = *(_QWORD *)v16;
  *(_QWORD *)(v16 + 8) = v15;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v16 = v17 | v18 & 7;
  *(_QWORD *)(v17 + 8) = v16;
  *v15 = v16 | *v15 & 7;
  v48.m128i_i64[0] = 0x10000000;
  v48.m128i_i32[2] = v47;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  sub_1E1A9C0(v16, v14, &v48);
  v48.m128i_i64[0] = 0;
  v49 = 0;
  v48.m128i_i32[2] = v46;
  v50 = 0;
  v51 = 0;
  sub_1E1A9C0(v16, v14, &v48);
  v19 = *(_DWORD *)(a3 + 24);
  if ( !v19 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_11;
  }
  v20 = *(_QWORD *)(a3 + 8);
  v21 = 1;
  result = 0;
  for ( i = (v19 - 1) & ((a2 >> 9) ^ (a2 >> 4)); ; i = (v19 - 1) & v26 )
  {
    v24 = (unsigned __int64 **)(v20 + 24LL * i);
    v25 = *v24;
    if ( (unsigned __int64 *)a2 != *v24 )
      break;
    if ( !*((_DWORD *)v24 + 2) )
      return result;
LABEL_7:
    v26 = v21 + i;
    ++v21;
  }
  if ( v25 )
    goto LABEL_7;
  v34 = *((_DWORD *)v24 + 2);
  if ( v34 != -1 )
  {
    if ( v34 == -2 && !result )
      result = (unsigned __int64 *)(v20 + 24LL * i);
    goto LABEL_7;
  }
  v35 = *(_DWORD *)(a3 + 16);
  if ( !result )
    result = (unsigned __int64 *)v24;
  ++*(_QWORD *)a3;
  v36 = v35 + 1;
  if ( 4 * v36 < 3 * v19 )
  {
    if ( v19 - *(_DWORD *)(a3 + 20) - v36 > v19 >> 3 )
      goto LABEL_25;
    sub_1FE7AA0(a3, v19);
    v37 = *(_DWORD *)(a3 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v40 = 1;
      for ( j = (v37 - 1) & ((a2 >> 9) ^ (a2 >> 4)); ; j = v38 & v42 )
      {
        v39 = *(_QWORD *)(a3 + 8);
        result = (unsigned __int64 *)(v39 + 24LL * j);
        if ( a2 == *result )
        {
          if ( !*((_DWORD *)result + 2) )
            goto LABEL_36;
        }
        else if ( !*result )
        {
          v44 = *((_DWORD *)result + 2);
          if ( v44 == -1 )
          {
            v36 = *(_DWORD *)(a3 + 16) + 1;
            if ( v25 )
              result = v25;
            goto LABEL_25;
          }
          if ( !v25 && v44 == -2 )
            v25 = (unsigned __int64 *)(v39 + 24LL * j);
        }
        v42 = v40 + j;
        ++v40;
      }
    }
LABEL_55:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_11:
  sub_1FE7AA0(a3, 2 * v19);
  v27 = *(_DWORD *)(a3 + 24);
  if ( !v27 )
    goto LABEL_55;
  v28 = v27 - 1;
  v30 = 1;
  v31 = 0;
  v32 = (v27 - 1) & ((a2 >> 9) ^ (a2 >> 4));
  while ( 2 )
  {
    v29 = *(_QWORD *)(a3 + 8);
    result = (unsigned __int64 *)(v29 + 24LL * v32);
    if ( a2 == *result )
    {
      if ( !*((_DWORD *)result + 2) )
      {
LABEL_36:
        v36 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_25;
      }
      goto LABEL_15;
    }
    if ( *result )
    {
LABEL_15:
      v33 = v30 + v32;
      ++v30;
      v32 = v28 & v33;
      continue;
    }
    break;
  }
  v43 = *((_DWORD *)result + 2);
  if ( v43 != -1 )
  {
    if ( v43 == -2 && !v31 )
      v31 = (unsigned __int64 *)(v29 + 24LL * v32);
    goto LABEL_15;
  }
  v36 = *(_DWORD *)(a3 + 16) + 1;
  if ( v31 )
    result = v31;
LABEL_25:
  *(_DWORD *)(a3 + 16) = v36;
  if ( *result || *((_DWORD *)result + 2) != -1 )
    --*(_DWORD *)(a3 + 20);
  *result = a2;
  *((_DWORD *)result + 2) = 0;
  *((_DWORD *)result + 4) = v47;
  return result;
}
