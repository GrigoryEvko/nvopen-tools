// Function: sub_1620350
// Address: 0x1620350
//
__int64 *__fastcall sub_1620350(__int64 *a1, __int64 a2)
{
  char v2; // r14
  char v3; // bl
  __int64 v4; // rax
  void *v5; // r13
  __int64 v6; // rax
  size_t v7; // rdx
  __int64 v8; // rax
  void *v9; // r9
  size_t v10; // rdx
  size_t v11; // r10
  __int64 *v12; // r15
  void *v13; // rsi
  size_t v14; // rdx
  __int64 v15; // rax
  void *v16; // rsi
  size_t v17; // rdx
  __int64 v18; // rax
  size_t v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rsi
  __m128i v24; // rax
  size_t v25; // rdx
  size_t v26; // [rsp+0h] [rbp-F0h]
  void *v27; // [rsp+0h] [rbp-F0h]
  __int64 v28; // [rsp+8h] [rbp-E8h]
  size_t v29; // [rsp+8h] [rbp-E8h]
  void *v30; // [rsp+8h] [rbp-E8h]
  __int32 v31; // [rsp+14h] [rbp-DCh]
  __int64 v32; // [rsp+18h] [rbp-D8h]
  __int64 v33; // [rsp+20h] [rbp-D0h]
  size_t v34; // [rsp+20h] [rbp-D0h]
  void *v35; // [rsp+20h] [rbp-D0h]
  size_t v36; // [rsp+28h] [rbp-C8h]
  size_t v37; // [rsp+28h] [rbp-C8h]
  __int64 v38; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-B8h]
  __m128i v40; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v41; // [rsp+60h] [rbp-90h]
  __m128i v42; // [rsp+80h] [rbp-70h] BYREF
  char v43; // [rsp+90h] [rbp-60h]
  __int64 v44; // [rsp+A8h] [rbp-48h]
  size_t v45; // [rsp+B0h] [rbp-40h]

  v2 = *(_BYTE *)(a2 + 56);
  if ( v2 )
  {
    v24.m128i_i64[0] = sub_161E970(*(_QWORD *)(a2 + 48));
    v3 = *(_BYTE *)(a2 + 40);
    v40 = v24;
    v33 = v24.m128i_i64[1];
    if ( !v3 )
      goto LABEL_3;
  }
  else
  {
    v3 = *(_BYTE *)(a2 + 40);
    if ( !v3 )
      goto LABEL_3;
  }
  v28 = sub_161E970(*(_QWORD *)(a2 + 32));
  v31 = *(_DWORD *)(a2 + 24);
  v26 = v25;
LABEL_3:
  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(void **)(a2 + 8 * (1 - v4));
  if ( v5 )
  {
    v6 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v4)));
    v36 = v7;
    v5 = (void *)v6;
    v4 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v36 = 0;
  }
  v8 = -v4;
  v9 = *(void **)(a2 + 8 * v8);
  if ( v9 )
  {
    v9 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * v8));
    v11 = v10;
  }
  else
  {
    v11 = 0;
  }
  v12 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v12 = (__int64 *)*v12;
  if ( v2 )
  {
    v40.m128i_i64[1] = v33;
    v41 = _mm_loadu_si128(&v40);
    if ( !v3 )
    {
      v16 = (void *)v41.m128i_i64[0];
      v41.m128i_i64[1] = v33;
      v17 = v33;
      if ( !v33 )
      {
        v38 = 0;
        v39 = 1;
        v43 = 0;
        goto LABEL_16;
      }
      goto LABEL_15;
    }
    v32 = 0;
    v44 = v28;
    v13 = (void *)v28;
    v45 = v26;
    v14 = v26;
    if ( !v26 )
    {
LABEL_13:
      if ( v2 )
      {
        v16 = (void *)v41.m128i_i64[0];
        v3 = v2;
        v41.m128i_i64[1] = v33;
        v17 = v33;
        if ( v33 )
        {
LABEL_15:
          v30 = v9;
          v34 = v11;
          v18 = sub_161FF10(v12, v16, v17);
          v43 = v3;
          v11 = v34;
          LOBYTE(v39) = 1;
          v9 = v30;
          v38 = v18;
          if ( !v3 )
            goto LABEL_16;
          goto LABEL_27;
        }
        v38 = 0;
        v39 = 1;
        v43 = 1;
LABEL_27:
        v42.m128i_i32[0] = v31;
        v42.m128i_i64[1] = v32;
        goto LABEL_16;
      }
LABEL_26:
      LOBYTE(v39) = 0;
      v43 = 1;
      goto LABEL_27;
    }
LABEL_12:
    v27 = v9;
    v29 = v11;
    v15 = sub_161FF10(v12, v13, v14);
    v9 = v27;
    v11 = v29;
    v32 = v15;
    goto LABEL_13;
  }
  if ( v3 )
  {
    v44 = v28;
    v13 = (void *)v28;
    v45 = v26;
    v14 = v26;
    if ( !v26 )
    {
      v32 = 0;
      goto LABEL_26;
    }
    goto LABEL_12;
  }
  LOBYTE(v39) = 0;
  v43 = 0;
LABEL_16:
  v19 = v36;
  v20 = 0;
  if ( v36 )
  {
    v35 = v9;
    v37 = v11;
    v21 = sub_161FF10(v12, v5, v19);
    v9 = v35;
    v11 = v37;
    v20 = v21;
  }
  v22 = 0;
  if ( v11 )
    v22 = sub_161FF10(v12, v9, v11);
  *a1 = sub_15BF650(v12, v22, v20, &v42, (__int64)&v38, 2u, 1);
  return a1;
}
